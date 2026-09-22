from __future__ import annotations

from collections import defaultdict
from typing import Any


class AWSResourceIntelligence:
    """Deterministic evidence-based intelligence over AWS resources.

    This layer deliberately does not invent resource-level AWS costs.
    Cost Explorer service-level cost is shown as service context only.
    Resource-level savings become monetary only when resource-level billing
    data is available.
    """

    def __init__(self, resources: list[Any], billing_records: list[Any]):
        self.resources = resources
        self.billing_records = billing_records
        self.service_costs = self._service_costs()

    def _service_costs(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for record in self.billing_records:
            service = getattr(record, "service", None) or "Unknown"
            totals[service] += float(getattr(record, "monthly_cost", 0) or 0)
        return {k: round(v, 2) for k, v in totals.items()}

    @staticmethod
    def _metadata(resource: Any) -> dict[str, Any]:
        value = getattr(resource, "metadata_json", None)
        return value if isinstance(value, dict) else {}

    def analyze(self) -> dict[str, Any]:
        recommendations: list[dict[str, Any]] = []
        healthy = 0

        for resource in self.resources:
            service = resource.service
            rid = resource.resource_id
            name = resource.resource_name or rid
            cpu = resource.cpu_utilization
            state = (resource.state or "").lower()
            metadata = self._metadata(resource)
            resource_monthly_cost = getattr(resource, "monthly_cost", None)
            cost_currency = getattr(resource, "cost_currency", None) or "USD"

            evidence: list[str] = []
            recommendation_type = None
            priority = "low"
            title = None
            action = None

            if service == "EC2":
                if state == "stopped":
                    recommendation_type = "scheduling"
                    priority = "high"
                    title = "Review stopped EC2 instance"
                    action = "Confirm whether this instance is still required; if not, remove or schedule it according to change policy."
                    evidence.append("EC2 state is stopped")
                elif cpu is not None and cpu < 10:
                    recommendation_type = "right_sizing"
                    priority = "high"
                    title = "Potential EC2 underutilization"
                    action = "Review the instance size and workload requirements before rightsizing."
                    evidence.append(f"Average CPU utilization is {cpu:.1f}%")
                elif cpu is not None and cpu < 20:
                    recommendation_type = "right_sizing"
                    priority = "medium"
                    title = "EC2 utilization is low"
                    action = "Review instance sizing and workload patterns before making a change."
                    evidence.append(f"Average CPU utilization is {cpu:.1f}%")
                else:
                    healthy += 1

            elif service == "EBS":
                attached = metadata.get("attached_instance_ids") or []
                state_is_available = state == "available"
                if state_is_available and not attached:
                    recommendation_type = "storage_cleanup"
                    priority = "high"
                    title = "Potentially unattached EBS volume"
                    action = "Verify the volume is no longer needed before deleting it."
                    evidence.append("EBS volume state is available")
                    evidence.append("No attached EC2 instance was detected")
                else:
                    healthy += 1

            elif service == "RDS":
                if state in {"stopped", "stopping"}:
                    recommendation_type = "scheduling"
                    priority = "medium"
                    title = "Review stopped RDS database"
                    action = "Confirm the database is intentionally stopped and review its ongoing cost/retention requirements."
                    evidence.append(f"RDS state is {state}")
                else:
                    healthy += 1

            elif service == "NAT Gateway":
                if state not in {"available", "pending"}:
                    recommendation_type = "nat_optimization"
                    priority = "medium"
                    title = "Review NAT Gateway state"
                    action = "Verify whether this NAT Gateway is still required and inspect traffic before changing it."
                    evidence.append(f"NAT Gateway state is {state or 'unknown'}")
                else:
                    healthy += 1

            else:
                healthy += 1

            estimated_saving = None
            saving_method = "Resource-level billing unavailable"
            if recommendation_type and resource_monthly_cost is not None and resource_monthly_cost > 0:
                rates = {
                    "right_sizing": 0.30,
                    "scheduling": 0.40,
                    "storage_cleanup": 0.80,
                    "nat_optimization": 0.15,
                }
                rate = rates.get(recommendation_type)
                if rate is not None:
                    estimated_saving = round(float(resource_monthly_cost) * rate, 2)
                    saving_method = f"{int(rate * 100)}% deterministic CloudSense savings model"
                    evidence.append(
                        f"Resource-level AWS cost is {resource_monthly_cost:.2f} {cost_currency}/month"
                    )

            if recommendation_type:
                recommendations.append({
                    "resource_id": rid,
                    "resource_name": name,
                    "service": service,
                    "resource_type": resource.resource_type,
                    "region": resource.region,
                    "state": resource.state,
                    "environment": resource.environment,
                    "team": resource.team,
                    "cpu_utilization": resource.cpu_utilization,
                    "memory_utilization": resource.memory_utilization,
                    "monthly_cost": resource_monthly_cost,
                    "cost_currency": cost_currency,
                    "recommendation_type": recommendation_type,
                    "priority": priority,
                    "title": title,
                    "action": action,
                    "evidence": evidence,
                    "service_cost_context": self.service_costs.get(service, 0),
                    "estimated_monthly_saving": estimated_saving,
                    "estimated_annual_saving": round(estimated_saving * 12, 2) if estimated_saving is not None else None,
                    "saving_currency": cost_currency if estimated_saving is not None else None,
                    "saving_status": "Available" if estimated_saving is not None else "Resource-level billing required for monetary savings estimate",
                    "saving_method": saving_method,
                    "safe_to_execute": False,
                })

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 9), x["service"], x["resource_id"] or "")
        )

        return {
            "resource_count": len(self.resources),
            "recommendation_count": len(recommendations),
            "healthy_or_non_flagged_count": healthy,
            "service_cost_context": self.service_costs,
            "recommendations": recommendations,
            "monetary_savings_note": (
                "Resource-level monetary savings are shown only when AWS Cost Explorer resource-level billing is available. "
                "Savings percentages are deterministic CloudSense estimates, not AWS pricing quotes."
            ),
        }
