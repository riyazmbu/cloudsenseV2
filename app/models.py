from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from .db import Base


class BillingRecord(Base):
    __tablename__ = "billing_records"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), default="AWS")
    billing_month = Column(String(30), nullable=True)
    service = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    resource_id = Column(String(150), nullable=True)
    environment = Column(String(50), nullable=True)
    team = Column(String(100), nullable=True)
    daily_cost = Column(Float, default=0)
    monthly_cost = Column(Float, default=0)
    cpu_utilization = Column(Float, nullable=True)
    memory_utilization = Column(Float, nullable=True)
    cost_currency = Column(String(10), nullable=True)
    hours_running = Column(Float, nullable=True)
    source_file = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BillingDocument(Base):
    __tablename__ = "billing_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(30), nullable=False)
    extracted_text = Column(Text, nullable=True)
    extraction_method = Column(String(50), nullable=True)
    ocr_required = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AWSResource(Base):
    __tablename__ = "aws_resources"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(20), nullable=False, index=True)
    service = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(300), nullable=True, index=True)
    resource_type = Column(String(150), nullable=True)
    resource_name = Column(String(300), nullable=True)
    region = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    environment = Column(String(100), nullable=True)
    team = Column(String(150), nullable=True)
    cpu_utilization = Column(Float, nullable=True)
    memory_utilization = Column(Float, nullable=True)
    monthly_cost = Column(Float, nullable=True)
    cost_currency = Column(String(10), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    collected_at = Column(DateTime, default=datetime.utcnow, index=True)


class CostSnapshot(Base):
    __tablename__ = "cost_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(30), nullable=False, index=True)
    account_id = Column(String(20), nullable=True, index=True)
    region = Column(String(100), nullable=True)
    monthly_cost = Column(Float, default=0)
    resource_count = Column(Integer, default=0)
    anomaly_count = Column(Integer, default=0)
    potential_monthly_saving = Column(Float, default=0)
    captured_at = Column(DateTime, default=datetime.utcnow, index=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    department = Column(String(150), nullable=True)
    team = Column(String(150), nullable=True)
    status = Column(String(30), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)


class ResourceOwnership(Base):
    __tablename__ = "resource_ownership"

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(String(300), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    usage_percentage = Column(Float, default=100.0)
    source = Column(String(50), default="MANUAL_MAPPING")
    enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProjectUserMapping(Base):
    __tablename__ = "project_user_mappings"

    id = Column(Integer, primary_key=True, index=True)
    project_key = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    usage_percentage = Column(Float, default=100.0)
    enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class WorkloadUsageSignal(Base):
    __tablename__ = "workload_usage_signals"

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(String(300), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    period_key = Column(String(100), nullable=False, index=True)
    metric = Column(String(100), nullable=True)
    usage_value = Column(Float, nullable=True)
    usage_percentage = Column(Float, nullable=False)
    evidence_json = Column(JSON, nullable=True)
    enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class CloudTrailActivity(Base):
    __tablename__ = "cloudtrail_activity"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(20), nullable=True, index=True)
    resource_id = Column(String(300), nullable=True, index=True)
    principal = Column(String(255), nullable=True, index=True)
    action = Column(String(255), nullable=True)
    event_time = Column(DateTime, nullable=True, index=True)
    event_source = Column(String(255), nullable=True)
    event_name = Column(String(255), nullable=True)
    raw_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserCostUsage(Base):
    __tablename__ = "user_cost_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    resource_id = Column(String(300), nullable=False, index=True)
    resource_type = Column(String(150), nullable=True)
    service = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    period_key = Column(String(100), nullable=False, index=True)
    period_start = Column(String(30), nullable=True)
    period_end = Column(String(30), nullable=True)
    resource_cost = Column(Float, default=0)
    usage_metric = Column(String(100), nullable=True)
    usage_value = Column(Float, nullable=True)
    usage_percentage = Column(Float, nullable=True)
    attributed_cost = Column(Float, default=0)
    attribution_method = Column(String(60), nullable=False)
    confidence = Column(String(30), nullable=True)
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ReconciliationResult(Base):
    __tablename__ = "reconciliation_results"

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(String(300), nullable=False, index=True)
    period_key = Column(String(100), nullable=False, index=True)
    resource_cost = Column(Float, default=0)
    attributed_cost = Column(Float, default=0)
    unattributed_cost = Column(Float, default=0)
    reconciled_total = Column(Float, default=0)
    difference = Column(Float, default=0)
    status = Column(String(30), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AttributionAudit(Base):
    __tablename__ = "attribution_audit"

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(String(300), nullable=False, index=True)
    period_key = Column(String(100), nullable=False, index=True)
    selected_method = Column(String(60), nullable=False)
    confidence = Column(String(30), nullable=True)
    evidence_json = Column(JSON, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
