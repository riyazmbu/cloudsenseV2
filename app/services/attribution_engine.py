from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models import (
    AWSResource,
    AttributionAudit,
    ProjectUserMapping,
    ReconciliationResult,
    ResourceOwnership,
    User,
    UserCostUsage,
    WorkloadUsageSignal,
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _tags(resource: AWSResource) -> dict[str, str]:
    metadata = resource.metadata_json or {}
    raw = metadata.get("tags") or {}
    return {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if k and v is not None}


def _first_tag(tags: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = tags.get(key.lower())
        if value:
            return value
    return None


def _get_or_create_user(db: Session, user_key: str, display_name: str | None = None) -> User:
    key = user_key.strip()
    user = db.query(User).filter(User.user_id == key).first()
    if not user:
        user = User(user_id=key, name=(display_name or key), status="active")
        db.add(user)
        db.flush()
    return user


def _validate_percentages(items: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], float]:
    valid = [(uid, float(pct)) for uid, pct in items if float(pct or 0) > 0]
    total = sum(p for _, p in valid)
    if total > 100.000001:
        raise ValueError(f"Attribution percentages exceed 100% ({total:.2f}%).")
    return valid, total


def _delete_period_results(db: Session, period_key: str) -> None:
    db.query(UserCostUsage).filter(UserCostUsage.period_key == period_key).delete(synchronize_session=False)
    db.query(ReconciliationResult).filter(ReconciliationResult.period_key == period_key).delete(synchronize_session=False)
    db.query(AttributionAudit).filter(AttributionAudit.period_key == period_key).delete(synchronize_session=False)


def calculate_attribution(db: Session, period_key: str = "current") -> dict[str, Any]:
    """Deterministic user/resource cost attribution.

    Priority:
      1. Explicit Owner tag
      2. Manual resource mapping
      3. Project/application mapping
      4. Workload usage signal
      5. CloudTrail can only support the audit trail; it never fabricates ownership
      6. Unattributed remainder
    """
    resources = db.query(AWSResource).all()
    _delete_period_results(db, period_key)

    user_totals: defaultdict[str, float] = defaultdict(float)
    resource_results: list[dict[str, Any]] = []
    total_resource_cost = 0.0
    total_attributed = 0.0
    total_unattributed = 0.0
    method_counts: defaultdict[str, int] = defaultdict(int)

    for resource in resources:
        cost = float(resource.monthly_cost or 0)
        if cost < 0:
            cost = 0.0
        total_resource_cost += cost
        rid = resource.resource_id or f"db:{resource.id}"
        tags = _tags(resource)
        owner = _first_tag(tags, "owner", "owner_id", "owner_user", "owneremail", "owner_email")
        project = _first_tag(tags, "project", "project_id", "application", "application_id")

        allocations: list[tuple[str, float, str, str, str | None]] = []
        evidence: list[str] = []

        if owner:
            user = _get_or_create_user(db, owner)
            allocations = [(user.user_id, 100.0, "OWNER_TAG", "HIGH", f"Owner tag = {owner}")]
            evidence.append(f"Owner tag matched user {owner}")
        else:
            manual = db.query(ResourceOwnership).filter(
                ResourceOwnership.resource_id == rid,
                ResourceOwnership.enabled == 1,
            ).all()
            if manual:
                pairs, total = _validate_percentages([(m.user_id, m.usage_percentage) for m in manual])
                for uid, pct in pairs:
                    _get_or_create_user(db, uid)
                    allocations.append((uid, pct, "MANUAL_MAPPING", "HIGH", f"Manual mapping {pct:.2f}%"))
                evidence.append("Explicit administrator resource mapping")
            elif project:
                mappings = db.query(ProjectUserMapping).filter(
                    ProjectUserMapping.project_key == project,
                    ProjectUserMapping.enabled == 1,
                ).all()
                if mappings:
                    pairs, total = _validate_percentages([(m.user_id, m.usage_percentage) for m in mappings])
                    for uid, pct in pairs:
                        _get_or_create_user(db, uid)
                        allocations.append((uid, pct, "PROJECT_MAPPING", "HIGH" if total >= 99.99 else "MEDIUM", f"Project {project} mapped {pct:.2f}%"))
                    evidence.append(f"Project/application mapping = {project}")
            if not allocations:
                signals = db.query(WorkloadUsageSignal).filter(
                    WorkloadUsageSignal.resource_id == rid,
                    WorkloadUsageSignal.period_key == period_key,
                    WorkloadUsageSignal.enabled == 1,
                ).all()
                if signals:
                    pairs, total = _validate_percentages([(s.user_id, s.usage_percentage) for s in signals])
                    for uid, pct in pairs:
                        _get_or_create_user(db, uid)
                        metric = next((x.metric for x in signals if x.user_id == uid and abs(float(x.usage_percentage or 0) - pct) < 0.0001), "usage")
                        allocations.append((uid, pct, "WORKLOAD_USAGE", "HIGH" if total >= 99.99 else "MEDIUM", f"{metric or 'usage'} = {pct:.2f}%"))
                    evidence.append("Workload/application usage telemetry")

        allocated_pct = min(100.0, sum(p for _, p, _, _, _ in allocations))
        unattributed_pct = max(0.0, 100.0 - allocated_pct)
        attributed_amount = 0.0

        for uid, pct, method, confidence, method_evidence in allocations:
            amount = round(cost * pct / 100.0, 8)
            attributed_amount += amount
            user_totals[uid] += amount
            method_counts[method] += 1
            db.add(UserCostUsage(
                user_id=uid,
                resource_id=rid,
                resource_type=resource.resource_type,
                service=resource.service,
                region=resource.region,
                period_key=period_key,
                period_start=None,
                period_end=None,
                resource_cost=cost,
                usage_metric="attribution_percentage",
                usage_value=pct,
                usage_percentage=pct,
                attributed_cost=amount,
                attribution_method=method,
                confidence=confidence,
                evidence=method_evidence,
                created_at=datetime.now(timezone.utc),
            ))

        unattributed_amount = round(max(0.0, cost - attributed_amount), 8)
        total_attributed += attributed_amount
        total_unattributed += unattributed_amount

        if not allocations:
            method = "UNATTRIBUTED"
            confidence = "N/A"
            method_counts[method] += 1
            evidence.append("No reliable ownership or usage allocation signal")
        elif unattributed_amount > 0:
            method = "PARTIAL_ALLOCATION"
            confidence = min((a[3] for a in allocations), key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x, 0))
        else:
            method = allocations[0][2] if len({a[2] for a in allocations}) == 1 else "COMPOSITE"
            confidence = min((a[3] for a in allocations), key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x, 0))

        reconciliation = round(attributed_amount + unattributed_amount, 8)
        difference = round(cost - reconciliation, 8)
        status = "MATCHED" if abs(difference) <= 0.01 else ("OVER_ALLOCATED" if difference < 0 else "UNDER_ALLOCATED")
        db.add(ReconciliationResult(
            resource_id=rid,
            period_key=period_key,
            resource_cost=cost,
            attributed_cost=round(attributed_amount, 8),
            unattributed_cost=unattributed_amount,
            reconciled_total=reconciliation,
            difference=difference,
            status=status,
            created_at=datetime.now(timezone.utc),
        ))
        db.add(AttributionAudit(
            resource_id=rid,
            period_key=period_key,
            selected_method=method,
            confidence=confidence,
            evidence_json={"tags": tags, "signals": evidence, "allocated_percentage": allocated_pct},
            note="CloudTrail activity is supporting evidence only and does not create ownership by itself.",
            created_at=datetime.now(timezone.utc),
        ))

        resource_results.append({
            "resource_id": rid,
            "service": resource.service,
            "resource_name": resource.resource_name,
            "resource_cost": round(cost, 2),
            "attributed_cost": round(attributed_amount, 2),
            "unattributed_cost": round(unattributed_amount, 2),
            "allocated_percentage": round(allocated_pct, 2),
            "method": method,
            "confidence": confidence,
            "evidence": evidence,
            "reconciliation": status,
        })

    db.commit()
    coverage = (total_attributed / total_resource_cost * 100.0) if total_resource_cost else 0.0
    return {
        "period_key": period_key,
        "resource_count": len(resources),
        "user_count": len(user_totals),
        "total_resource_cost": round(total_resource_cost, 2),
        "total_attributed_cost": round(total_attributed, 2),
        "total_unattributed_cost": round(total_unattributed, 2),
        "attribution_coverage": round(coverage, 2),
        "reconciliation_difference": round(total_resource_cost - (total_attributed + total_unattributed), 8),
        "method_counts": dict(method_counts),
        "users": sorted([
            {"user_id": uid, "attributed_cost": round(cost, 2)}
            for uid, cost in user_totals.items()
        ], key=lambda x: x["attributed_cost"], reverse=True),
        "resources": resource_results,
    }
