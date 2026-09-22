from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import AWSResource, ProjectUserMapping, WorkloadUsageSignal, ReconciliationResult
from app.services.attribution_engine import calculate_attribution


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_owner_tag_gets_full_cost_and_reconciles():
    db = make_db()
    db.add(AWSResource(
        account_id="123456789012", service="EC2", resource_id="i-owner",
        resource_type="t3.small", region="ap-south-1", monthly_cost=100,
        metadata_json={"tags": {"Owner": "riyaz"}},
    ))
    db.commit()

    result = calculate_attribution(db)

    assert result["total_attributed_cost"] == 100
    assert result["total_unattributed_cost"] == 0
    assert result["reconciliation_difference"] == 0
    assert result["users"][0]["user_id"] == "riyaz"


def test_project_and_workload_mapping_allocate_shared_cost():
    db = make_db()
    db.add_all([
        AWSResource(account_id="1", service="EC2", resource_id="i-project", region="x", monthly_cost=100, metadata_json={"tags": {"Project": "A"}}),
        AWSResource(account_id="1", service="EC2", resource_id="i-work", region="x", monthly_cost=200, metadata_json={"tags": {}}),
    ])
    db.add_all([
        ProjectUserMapping(project_key="A", user_id="u1", usage_percentage=70),
        ProjectUserMapping(project_key="A", user_id="u2", usage_percentage=30),
        WorkloadUsageSignal(resource_id="i-work", user_id="u1", period_key="current", metric="requests", usage_percentage=60),
        WorkloadUsageSignal(resource_id="i-work", user_id="u2", period_key="current", metric="requests", usage_percentage=40),
    ])
    db.commit()

    result = calculate_attribution(db)
    totals = {x["user_id"]: x["attributed_cost"] for x in result["users"]}

    assert totals == {"u1": 190.0, "u2": 110.0}
    assert result["total_unattributed_cost"] == 0
    assert result["reconciliation_difference"] == 0
    assert all(r.status == "MATCHED" for r in db.query(ReconciliationResult).all())


def test_missing_signal_remains_unattributed():
    db = make_db()
    db.add(AWSResource(account_id="1", service="RDS", resource_id="db-1", region="x", monthly_cost=250, metadata_json={"tags": {}}))
    db.commit()

    result = calculate_attribution(db)

    assert result["total_attributed_cost"] == 0
    assert result["total_unattributed_cost"] == 250
    assert result["attribution_coverage"] == 0
    assert result["reconciliation_difference"] == 0
