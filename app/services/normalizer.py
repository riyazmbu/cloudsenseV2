import math

ALIASES = {
    "provider": ["provider", "cloud_provider", "cloud"],
    "billing_month": ["billing_month", "month", "billing period"],
    "service": ["service", "service_name", "product"],
    "region": ["region", "location"],
    "resource_id": ["resource_id", "resource", "instance_id", "id"],
    "environment": ["environment", "env"],
    "team": ["team", "owner", "department"],
    "daily_cost": ["daily_cost", "daily cost", "cost_per_day"],
    "monthly_cost": ["monthly_cost", "monthly cost", "cost_per_month"],
    "cpu_utilization": ["cpu_utilization", "cpu", "cpu_%", "cpu_percent"],
    "memory_utilization": ["memory_utilization", "memory", "memory_%", "memory_percent"],
    "hours_running": ["hours_running", "running_hours", "uptime_hours"],
}

def clean_key(value):
    return str(value).strip().lower().replace("-", "_")

def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value).strip()

def find_value(row, aliases):
    normalized = {clean_key(k): v for k, v in row.items()}
    for alias in aliases:
        key = clean_key(alias)
        if key in normalized:
            return normalized[key]
    return None

def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").replace("₹", "").replace("%", "").strip())
    except ValueError:
        return None

def normalize_row(row):
    daily = to_float(find_value(row, ALIASES["daily_cost"]))
    monthly = to_float(find_value(row, ALIASES["monthly_cost"]))

    # Normalization rule: if only daily cost exists, estimate a 30-day monthly cost.
    if monthly is None and daily is not None:
        monthly = daily * 30

    return {
        "provider": clean_value(find_value(row, ALIASES["provider"])) or "AWS",
        "billing_month": clean_value(find_value(row, ALIASES["billing_month"])),
        "service": clean_value(find_value(row, ALIASES["service"])),
        "region": clean_value(find_value(row, ALIASES["region"])),
        "resource_id": clean_value(find_value(row, ALIASES["resource_id"])),
        "environment": clean_value(find_value(row, ALIASES["environment"])),
        "team": clean_value(find_value(row, ALIASES["team"])),
        "daily_cost": daily or 0,
        "monthly_cost": monthly or 0,
        "cpu_utilization": to_float(find_value(row, ALIASES["cpu_utilization"])),
        "memory_utilization": to_float(find_value(row, ALIASES["memory_utilization"])),
        "hours_running": to_float(find_value(row, ALIASES["hours_running"])),
    }

def normalize_records(rows):
    return [normalize_row(row) for row in rows]
