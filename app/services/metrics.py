from typing import Optional


def utilization_status(cpu: Optional[float], memory: Optional[float]) -> str:
    cpu = 0 if cpu is None else cpu
    memory = 0 if memory is None else memory

    if cpu < 10 and memory < 20:
        return "severely_underutilized"
    if cpu < 20 and memory < 30:
        return "underutilized"
    if cpu < 40 and memory < 50:
        return "moderately_utilized"
    return "healthy"


def calculate_health_score(cpu: Optional[float], memory: Optional[float], hours: Optional[float]) -> int:
    cpu = 0 if cpu is None else max(0, min(cpu, 100))
    memory = 0 if memory is None else max(0, min(memory, 100))
    hours = 0 if hours is None else max(0, min(hours, 24))
    utilization = (cpu + memory) / 2
    utilization_score = min(utilization * 1.5, 75)
    runtime_score = (hours / 24) * 25
    score = utilization_score + runtime_score
    return max(0, min(100, round(score)))


def analyze_record(record):
    service = (record.service or "").strip()
    service_key = service.lower()
    status = utilization_status(record.cpu_utilization, record.memory_utilization)
    score = calculate_health_score(record.cpu_utilization, record.memory_utilization, record.hours_running)

    reasons = []
    candidate = False
    recommendation_type = None

    cpu = record.cpu_utilization
    memory = record.memory_utilization
    hours = record.hours_running or 0
    environment = (record.environment or "").upper()
    resource_id = (record.resource_id or "").lower()
    monthly = float(record.monthly_cost or 0)

    # Compute services: CPU/memory evidence is appropriate.
    if service_key == "ec2":
        c = cpu or 0
        m = memory or 0
        # For non-production workloads, scheduling is considered before
        # right-sizing when the resource runs for most of the day.
        if environment in {"DEV", "QA", "TEST", "STAGING"} and hours >= 20:
            candidate = True
            recommendation_type = "scheduling"
            reasons.append("Non-production resource is running almost continuously")
        elif c < 10 and m < 20:
            candidate = True
            recommendation_type = "right_sizing"
            reasons.append("Very low CPU and memory utilization")
        elif c < 20 and m < 30:
            candidate = True
            recommendation_type = "right_sizing"
            reasons.append("Low average CPU and memory utilization")

    elif service_key == "rds":
        c = cpu or 0
        m = memory or 0
        # Non-production RDS with long runtime is primarily a scheduling
        # candidate. Production RDS can be evaluated for right-sizing.
        if environment in {"DEV", "QA", "TEST", "STAGING"} and hours >= 20:
            candidate = True
            recommendation_type = "scheduling"
            reasons.append("Non-production RDS is running for most of the day")
        elif c < 10 and m < 35:
            candidate = True
            recommendation_type = "rds_right_sizing"
            reasons.append("Low sustained RDS CPU and memory utilization")

    # Storage/network/serverless/CDN services do NOT use CPU/memory right-sizing.
    elif service_key == "ebs":
        if any(token in resource_id for token in ("unused", "unattached", "orphan")):
            candidate = True
            recommendation_type = "storage_cleanup"
            reasons.append("Billing record identifies the EBS volume as unused/unattached")
        # Otherwise there is not enough billing-only evidence to recommend deletion.

    elif service_key == "nat gateway":
        if monthly >= 250:
            candidate = True
            recommendation_type = "nat_optimization"
            reasons.append("NAT Gateway cost warrants traffic, routing and VPC endpoint review")
            reasons.append("CPU/memory metrics are not used for NAT Gateway optimization")

    elif service_key == "s3":
        if monthly >= 200:
            candidate = True
            recommendation_type = "s3_lifecycle"
            reasons.append("S3 spend warrants storage-class and lifecycle-policy review")

    elif service_key == "lambda":
        if monthly >= 100:
            candidate = True
            recommendation_type = "lambda_optimization"
            reasons.append("Lambda spend warrants duration, memory and invocation-efficiency review")

    elif service_key == "cloudfront":
        if monthly >= 150:
            candidate = True
            recommendation_type = "cloudfront_optimization"
            reasons.append("CloudFront spend warrants cache-hit, origin-transfer and distribution review")

    action_map = {
        "right_sizing": "Review compatible instance types, validate peak workload and load-test before approval.",
        "rds_right_sizing": "Review DB class, memory, connections, IOPS and SLA before testing a smaller class.",
        "scheduling": "Confirm owner, approved operating window and dependencies before scheduling stop/start.",
        "storage_cleanup": "Verify attachment/dependency, retention and backup requirements before cleanup.",
        "nat_optimization": "Review traffic, cross-AZ routing and VPC endpoints; do not use CPU/memory right-sizing.",
        "s3_lifecycle": "Review object age, access pattern, retention and lifecycle/storage-class policy.",
        "lambda_optimization": "Review duration, memory setting, invocation volume and workload latency before tuning.",
        "cloudfront_optimization": "Review cache-hit ratio, origin transfer and distribution behavior before changes.",
    }
    evidence_quality = "High" if len(reasons) >= 2 else ("Medium" if reasons else "Needs validation")
    return {
        "resource_id": record.resource_id,
        "service": record.service,
        "region": record.region,
        "environment": record.environment,
        "team": record.team,
        "monthly_cost": round(monthly, 2),
        "cpu_utilization": record.cpu_utilization,
        "memory_utilization": record.memory_utilization,
        "hours_running": record.hours_running,
        "utilization_status": status,
        "health_score": score,
        "optimization_candidate": candidate,
        "recommendation_type": recommendation_type,
        "reasons": reasons,
        "evidence": reasons,
        "evidence_quality": evidence_quality,
        "next_action": action_map.get(recommendation_type, "Validate the recommendation against current workload and change policy."),
        "safe_to_execute": False,
    }
