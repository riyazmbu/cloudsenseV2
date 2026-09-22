from typing import Optional

RIGHT_SIZE_SAVING_RATE = 0.30
SCHEDULE_SAVING_RATE = 0.40
STORAGE_CLEANUP_RATE = 0.80
RDS_OPTIMIZATION_RATE = 0.20
NAT_OPTIMIZATION_RATE = 0.15
S3_LIFECYCLE_RATE = 0.20
LAMBDA_OPTIMIZATION_RATE = 0.15
CLOUDFRONT_OPTIMIZATION_RATE = 0.10


def estimate_saving(monthly_cost: float, recommendation_type: str) -> float:
    rates = {
        "right_sizing": RIGHT_SIZE_SAVING_RATE,
        "rds_right_sizing": RDS_OPTIMIZATION_RATE,
        "scheduling": SCHEDULE_SAVING_RATE,
        "storage_cleanup": STORAGE_CLEANUP_RATE,
        "nat_optimization": NAT_OPTIMIZATION_RATE,
        "s3_lifecycle": S3_LIFECYCLE_RATE,
        "lambda_optimization": LAMBDA_OPTIMIZATION_RATE,
        "cloudfront_optimization": CLOUDFRONT_OPTIMIZATION_RATE,
    }
    return round(max(monthly_cost, 0) * rates.get(recommendation_type, 0), 2)


def pricing_for_metric(metric: dict) -> dict:
    monthly = metric.get("monthly_cost") or 0
    rec_type = metric.get("recommendation_type")
    if not metric.get("optimization_candidate"):
        return {
            **metric,
            "estimated_monthly_saving": 0,
            "estimated_yearly_saving": 0,
            "projected_monthly_cost": round(monthly, 2),
            "saving_rate": 0,
        }
    saving = estimate_saving(monthly, rec_type)
    return {
        **metric,
        "estimated_monthly_saving": saving,
        "estimated_yearly_saving": round(saving * 12, 2),
        "projected_monthly_cost": round(monthly - saving, 2),
        "saving_rate": round((saving / monthly) * 100, 2) if monthly else 0,
        "saving_status": "Available from current dataset" if saving else "No monetary estimate available",
        "safe_to_execute": False,
    }
