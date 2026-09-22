from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import boto3


class AWSCloudSenseCollector:
    """Read-only AWS collector for billing, resources and utilization metrics."""

    def __init__(self, access_key_id: str | None = None, secret_access_key: str | None = None, region: str = "ap-south-1", session=None):
        self.access_key_id = (access_key_id or "").strip()
        self.secret_access_key = (secret_access_key or "").strip()
        self.region = region.strip()
        self.session = session or boto3.Session(
            aws_access_key_id=self.access_key_id or None,
            aws_secret_access_key=self.secret_access_key or None,
            region_name=self.region,
        )

    def identity(self) -> dict[str, Any]:
        return self.session.client("sts", region_name=self.region).get_caller_identity()

    def billing(self, start: str | None = None, end: str | None = None):
        now = datetime.now(timezone.utc)
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else (now - timedelta(days=30)).date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else now.date()
        ce = self.session.client("ce", region_name="us-east-1")

        rows = []
        next_token = None
        while True:
            params = {
                "TimePeriod": {"Start": start_date.isoformat(), "End": end_date.isoformat()},
                "Granularity": "DAILY",
                "Metrics": ["UnblendedCost"],
                "GroupBy": [{"Type": "DIMENSION", "Key": "SERVICE"}],
            }
            if next_token:
                params["NextPageToken"] = next_token

            response = ce.get_cost_and_usage(**params)
            for result in response.get("ResultsByTime", []):
                day = result.get("TimePeriod", {}).get("Start")
                for group in result.get("Groups", []):
                    service_raw = (group.get("Keys") or ["Unknown"])[0]
                    cost_data = group.get("Metrics", {}).get("UnblendedCost", {})
                    amount = float(cost_data.get("Amount", 0))
                    rows.append({
                        "provider": "AWS",
                        "billing_date": day,
                        "billing_month": day[:7] if day else None,
                        "service": self._normalize_service(service_raw),
                        "aws_service": service_raw,
                        "region": self.region,
                        "resource_id": None,
                        "environment": None,
                        "team": None,
                        "daily_cost": round(amount, 6),
                        "monthly_cost": round(amount, 6),
                        "currency": cost_data.get("Unit", "USD"),
                        "source": "aws_cost_explorer",
                    })

            next_token = response.get("NextPageToken")
            if not next_token:
                break
        return rows

    def resource_level_billing(self, days: int = 14):
        """Collect EC2 resource-level costs when Cost Explorer resource billing is enabled.

        AWS limits this operation to the last 14 days and requires an EC2 service
        filter plus RESOURCE_ID grouping/filtering. The feature is opt-in in
        Cost Explorer Settings, so failure is returned as a capability message
        rather than making the whole AWS collection fail.
        """
        now = datetime.now(timezone.utc).date()
        days = min(max(days, 1), 14)
        start_date = now - timedelta(days=days)
        end_date = now

        ce = self.session.client("ce", region_name="us-east-1")
        rows = []
        next_token = None

        while True:
            params = {
                "TimePeriod": {
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                "Granularity": "DAILY",
                "Metrics": ["UnblendedCost"],
                "Filter": {
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": ["Amazon Elastic Compute Cloud - Compute"],
                        "MatchOptions": ["EQUALS"],
                    }
                },
                "GroupBy": [
                    {"Type": "DIMENSION", "Key": "RESOURCE_ID"}
                ],
            }
            if next_token:
                params["NextPageToken"] = next_token

            response = ce.get_cost_and_usage_with_resources(**params)

            for result in response.get("ResultsByTime", []):
                day = result.get("TimePeriod", {}).get("Start")
                estimated = result.get("Estimated", False)
                for group in result.get("Groups", []):
                    resource_id = (group.get("Keys") or [None])[0]
                    metric = group.get("Metrics", {}).get("UnblendedCost", {})
                    amount = float(metric.get("Amount", 0) or 0)
                    if not resource_id:
                        continue
                    rows.append({
                        "resource_id": resource_id,
                        "billing_date": day,
                        "daily_cost": round(amount, 6),
                        "currency": metric.get("Unit", "USD"),
                        "estimated": bool(estimated),
                    })

            next_token = response.get("NextPageToken")
            if not next_token:
                break

        totals = {}
        for row in rows:
            rid = row["resource_id"]
            totals.setdefault(rid, {"cost": 0.0, "currency": row["currency"], "days": 0, "estimated": False})
            totals[rid]["cost"] += row["daily_cost"]
            totals[rid]["days"] += 1
            totals[rid]["estimated"] = totals[rid]["estimated"] or row["estimated"]

        normalized = []
        for rid, item in totals.items():
            observed_days = max(item["days"], 1)
            monthly = item["cost"] * (30.0 / observed_days)
            normalized.append({
                "resource_id": rid,
                "observed_cost": round(item["cost"], 6),
                "observed_days": observed_days,
                "monthly_cost": round(item["cost"], 2),
                "currency": item["currency"],
                "cost_period_days": observed_days,
                "cost_is_estimate": bool(item["estimated"]),
                "estimated": item["estimated"],
                "source": "aws_cost_explorer_resource_level",
            })

        return {
            "enabled": True,
            "days": days,
            "rows": normalized,
            "error": None,
        }

    @staticmethod
    def _tags(resource_tags: list[dict[str, Any]] | None) -> dict[str, str]:
        return {
            t.get("Key"): t.get("Value")
            for t in (resource_tags or [])
            if t.get("Key")
        }

    def ec2_instances(self):
        ec2 = self.session.client("ec2", region_name=self.region)
        resources = []
        for page in ec2.get_paginator("describe_instances").paginate():
            for reservation in page.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    tags = self._tags(instance.get("Tags"))
                    resources.append({
                        "provider": "AWS",
                        "service": "EC2",
                        "resource_id": instance.get("InstanceId"),
                        "resource_type": instance.get("InstanceType"),
                        "region": self.region,
                        "environment": tags.get("Environment") or tags.get("environment") or tags.get("Env") or tags.get("env"),
                        "team": tags.get("Team") or tags.get("team"),
                        "resource_name": tags.get("Name") or tags.get("name") or instance.get("InstanceId"),
                        "state": instance.get("State", {}).get("Name"),
                        "launch_time": instance.get("LaunchTime").isoformat() if instance.get("LaunchTime") else None,
                        "availability_zone": instance.get("Placement", {}).get("AvailabilityZone"),
                        "tags": tags,
                    })
        return resources

    def ebs_volumes(self):
        ec2 = self.session.client("ec2", region_name=self.region)
        resources = []
        for page in ec2.get_paginator("describe_volumes").paginate():
            for volume in page.get("Volumes", []):
                tags = self._tags(volume.get("Tags"))
                attachments = volume.get("Attachments", [])
                resources.append({
                    "provider": "AWS",
                    "service": "EBS",
                    "resource_id": volume.get("VolumeId"),
                    "resource_type": volume.get("VolumeType"),
                    "region": self.region,
                    "environment": tags.get("Environment") or tags.get("environment"),
                    "team": tags.get("Team") or tags.get("team"),
                    "resource_name": tags.get("Name") or tags.get("name") or volume.get("VolumeId"),
                    "state": volume.get("State"),
                    "size_gb": volume.get("Size"),
                    "iops": volume.get("Iops"),
                    "attached_instance_ids": [a.get("InstanceId") for a in attachments if a.get("InstanceId")],
                    "tags": tags,
                })
        return resources

    def _rds_tags(self, resource_arn: str | None):
        if not resource_arn:
            return {}
        try:
            rds = self.session.client("rds", region_name=self.region)
            response = rds.list_tags_for_resource(ResourceName=resource_arn)
            return {t.get("Key"): t.get("Value") for t in response.get("TagList", []) if t.get("Key")}
        except Exception:
            return {}

    def _lambda_tags(self, resource_arn: str | None):
        if not resource_arn:
            return {}
        try:
            response = self.session.client("lambda", region_name=self.region).list_tags(Resource=resource_arn)
            return response.get("Tags") or {}
        except Exception:
            return {}

    def _s3_tags(self, bucket_name: str | None):
        if not bucket_name:
            return {}
        try:
            response = self.session.client("s3", region_name=self.region).get_bucket_tagging(Bucket=bucket_name)
            return {t.get("Key"): t.get("Value") for t in response.get("TagSet", []) if t.get("Key")}
        except Exception:
            return {}

    def _cloudfront_tags(self, resource_arn: str | None):
        if not resource_arn:
            return {}
        try:
            response = self.session.client("cloudfront", region_name="us-east-1").list_tags_for_resource(Resource=resource_arn)
            return {t.get("Key"): t.get("Value") for t in response.get("Tags", {}).get("Items", []) if t.get("Key")}
        except Exception:
            return {}

    def rds_instances(self):
        rds = self.session.client("rds", region_name=self.region)
        resources = []
        try:
            for page in rds.get_paginator("describe_db_instances").paginate():
                for db in page.get("DBInstances", []):
                    resources.append({
                        "provider": "AWS",
                        "service": "RDS",
                        "resource_id": db.get("DBInstanceArn") or db.get("DBInstanceIdentifier"),
                        "resource_type": db.get("DBInstanceClass"),
                        "region": self.region,
                        "resource_name": db.get("DBInstanceIdentifier"),
                        "state": db.get("DBInstanceStatus"),
                        "engine": db.get("Engine"),
                        "engine_version": db.get("EngineVersion"),
                        "storage_gb": db.get("AllocatedStorage"),
                        "availability_zone": db.get("AvailabilityZone"),
                    })
        except Exception as exc:
            return {"resources": [], "error": str(exc)}
        return resources

    def s3_buckets(self):
        s3 = self.session.client("s3", region_name=self.region)
        resources = []
        try:
            response = s3.list_buckets()
            for bucket in response.get("Buckets", []):
                name = bucket.get("Name")
                location = "unknown"
                try:
                    location = s3.get_bucket_location(Bucket=name).get("LocationConstraint") or "us-east-1"
                except Exception:
                    pass
                resources.append({
                    "provider": "AWS",
                    "service": "S3",
                    "resource_id": name,
                    "resource_type": "Bucket",
                    "region": location,
                    "resource_name": name,
                    "state": "active",
                    "created_at": bucket.get("CreationDate").isoformat() if bucket.get("CreationDate") else None,
                })
        except Exception as exc:
            return {"resources": [], "error": str(exc)}
        return resources

    def lambda_functions(self):
        lam = self.session.client("lambda", region_name=self.region)
        resources = []
        try:
            for page in lam.get_paginator("list_functions").paginate():
                for fn in page.get("Functions", []):
                    resources.append({
                        "provider": "AWS",
                        "service": "Lambda",
                        "resource_id": fn.get("FunctionArn") or fn.get("FunctionName"),
                        "resource_type": "Function",
                        "region": self.region,
                        "resource_name": fn.get("FunctionName"),
                        "state": "active",
                        "runtime": fn.get("Runtime"),
                        "memory_mb": fn.get("MemorySize"),
                        "timeout_seconds": fn.get("Timeout"),
                        "code_size": fn.get("CodeSize"),
                        "tags": self._lambda_tags(fn.get("FunctionArn")),
                    })
        except Exception as exc:
            return {"resources": [], "error": str(exc)}
        return resources

    def nat_gateways(self):
        ec2 = self.session.client("ec2", region_name=self.region)
        resources = []
        try:
            for page in ec2.get_paginator("describe_nat_gateways").paginate():
                for nat in page.get("NatGateways", []):
                    tags = self._tags(nat.get("Tags"))
                    resources.append({
                        "provider": "AWS",
                        "service": "NAT Gateway",
                        "resource_id": nat.get("NatGatewayId"),
                        "resource_type": nat.get("ConnectivityType"),
                        "region": self.region,
                        "resource_name": tags.get("Name") or tags.get("name") or nat.get("NatGatewayId"),
                        "state": nat.get("State"),
                        "subnet_id": nat.get("SubnetId"),
                        "vpc_id": nat.get("VpcId"),
                        "tags": tags,
                    })
        except Exception as exc:
            return {"resources": [], "error": str(exc)}
        return resources

    def cloudfront_distributions(self):
        cf = self.session.client("cloudfront", region_name="us-east-1")
        resources = []
        try:
            paginator = cf.get_paginator("list_distributions")
            for page in paginator.paginate():
                items = page.get("DistributionList", {}).get("Items", [])
                for dist in items:
                    resources.append({
                        "provider": "AWS",
                        "service": "CloudFront",
                        "resource_id": dist.get("ARN") or dist.get("Id"),
                        "resource_type": "Distribution",
                        "region": "global",
                        "resource_name": dist.get("DomainName") or dist.get("Id"),
                        "state": "Deployed" if dist.get("Status") == "Deployed" else dist.get("Status"),
                        "enabled": dist.get("Enabled"),
                        "domain_name": dist.get("DomainName"),
                        "tags": self._cloudfront_tags(dist.get("ARN") or dist.get("Id")),
                    })
        except Exception as exc:
            return {"resources": [], "error": str(exc)}
        return resources

    def all_resources(self):
        """Collect supported resource inventory without making one optional service failure fatal."""
        collections = {}
        errors = {}

        collectors = {
            "EC2": self.ec2_instances,
            "EBS": self.ebs_volumes,
            "RDS": self.rds_instances,
            "S3": self.s3_buckets,
            "Lambda": self.lambda_functions,
            "NAT Gateway": self.nat_gateways,
            "CloudFront": self.cloudfront_distributions,
        }

        for service, fn in collectors.items():
            try:
                result = fn()
                if isinstance(result, dict) and "resources" in result:
                    collections[service] = result["resources"]
                    if result.get("error"):
                        errors[service] = result["error"]
                else:
                    collections[service] = result
            except Exception as exc:
                collections[service] = []
                errors[service] = str(exc)

        resources = []
        for service_rows in collections.values():
            resources.extend(service_rows)

        return {
            "resources": resources,
            "by_service": collections,
            "errors": errors,
        }

    def ec2_cpu_metrics(self, instances, days: int = 7):
        """Collect EC2 CPU metrics for the full inventory in AWS API-sized batches."""
        cw = self.session.client("cloudwatch", region_name=self.region)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=min(days, 14))
        output = []
        valid_instances = [x for x in instances if x.get("resource_id")]

        # CloudWatch GetMetricData accepts a finite number of queries per call.
        # Batch the inventory instead of silently limiting the collection to 500 instances.
        for batch_start in range(0, len(valid_instances), 500):
            batch = valid_instances[batch_start:batch_start + 500]
            queries = []
            instance_indexes = {}
            for i, inst in enumerate(batch):
                rid = inst["resource_id"]
                query_id = f"cpu{i}"
                instance_indexes[query_id] = inst
                queries.append({
                    "Id": query_id,
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/EC2",
                            "MetricName": "CPUUtilization",
                            "Dimensions": [{"Name": "InstanceId", "Value": rid}],
                        },
                        "Period": 3600,
                        "Stat": "Average",
                    },
                    "ReturnData": True,
                })

            if not queries:
                continue

            response = cw.get_metric_data(
                MetricDataQueries=queries,
                StartTime=start,
                EndTime=end,
                ScanBy="TimestampDescending",
                MaxDatapoints=100800,
            )

            for result in response.get("MetricDataResults", []):
                values = [float(v) for v in result.get("Values", []) if v is not None]
                inst = instance_indexes.get(result.get("Id"))
                if values and inst:
                    output.append({
                        "resource_id": inst["resource_id"],
                        "metric": "CPUUtilization",
                        "average": round(sum(values) / len(values), 2),
                        "minimum": round(min(values), 2),
                        "maximum": round(max(values), 2),
                        "sample_count": len(values),
                        "period_days": min(days, 14),
                    })
        return output


    def cloudtrail_activity(self, resource_ids=None, days: int = 7, max_resources: int = 50):
        """Collect recent identity/activity signals for selected resources.

        CloudTrail is treated as supporting evidence only; this method never
        assigns cost by itself.
        """
        client = self.session.client("cloudtrail", region_name=self.region)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=min(max(days, 1), 90))
        rows = []
        for rid in list(resource_ids or [])[:max_resources]:
            if not rid:
                continue
            token = None
            while True:
                params = {"LookupAttributes": [{"AttributeKey": "ResourceName", "AttributeValue": rid}], "StartTime": start, "EndTime": end, "MaxResults": 50}
                if token:
                    params["NextToken"] = token
                try:
                    response = client.lookup_events(**params)
                except Exception as exc:
                    return {"rows": rows, "error": str(exc)}
                for event in response.get("Events", []):
                    cloud = event.get("CloudTrailEvent")
                    try:
                        import json
                        parsed = json.loads(cloud) if isinstance(cloud, str) else (cloud or {})
                    except Exception:
                        parsed = {}
                    principal = (parsed.get("userIdentity") or {}).get("arn") or (parsed.get("userIdentity") or {}).get("principalId")
                    rows.append({
                        "resource_id": rid,
                        "principal": principal,
                        "event_name": event.get("EventName"),
                        "event_source": event.get("EventSource"),
                        "event_time": event.get("EventTime").isoformat() if event.get("EventTime") else None,
                        "raw_json": parsed,
                    })
                token = response.get("NextToken")
                if not token:
                    break
        return {"rows": rows, "error": None}

    def collect_all(self, days: int = 30):
        identity = self.identity()
        instances = self.ec2_instances()
        cpu = self.ec2_cpu_metrics(instances, days=min(days, 14))
        cpu_map = {x["resource_id"]: x for x in cpu}

        for inst in instances:
            metric = cpu_map.get(inst["resource_id"])
            inst["cpu_utilization"] = metric["average"] if metric else None
            inst["memory_utilization"] = None

        billing = self.billing()
        resource_data = self.all_resources()

        resource_billing = {"enabled": False, "days": min(days, 14), "rows": [], "error": None}
        try:
            resource_billing = self.resource_level_billing(days=min(days, 14))
        except Exception as exc:
            resource_billing["error"] = str(exc)

        resource_level_rows = resource_billing.get("rows", [])

        cloudtrail = {"rows": [], "error": None}
        try:
            cloudtrail = self.cloudtrail_activity([r.get("resource_id") for r in resource_data["resources"]], days=min(days, 7), max_resources=50)
        except Exception as exc:
            cloudtrail = {"rows": [], "error": str(exc)}

        total_cost = sum(float(x.get("daily_cost", 0)) for x in billing)

        return {
            "account_id": identity.get("Account"),
            "region": self.region,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "billing": billing,
            "ec2": instances,
            "resources": resource_data["resources"],
            "resources_by_service": resource_data["by_service"],
            "resource_errors": resource_data["errors"],
            "cloudwatch": cpu,
            "resource_level_billing": resource_level_rows,
            "resource_level_billing_status": {
                "enabled": bool(resource_billing.get("enabled")),
                "days": resource_billing.get("days"),
                "error": resource_billing.get("error"),
            },
            "cloudtrail": cloudtrail.get("rows", []),
            "cloudtrail_status": {"enabled": bool(cloudtrail.get("rows")), "error": cloudtrail.get("error")},
            "summary": {
                "billing_rows": len(billing),
                "ec2_instances": len(instances),
                "resource_count": len(resource_data["resources"]),
                "cpu_metrics": len(cpu),
                "services": sorted({r.get("service") for r in resource_data["resources"] if r.get("service")}),
                "billing_total": round(total_cost, 6),
                "currency": billing[0].get("currency", "USD") if billing else "USD",
            },
        }

    @staticmethod
    def _normalize_service(service: str) -> str:
        s = (service or "").lower()
        if "elastic compute" in s or "ec2" in s:
            return "EC2"
        if "relational database" in s or "rds" in s:
            return "RDS"
        if "simple storage" in s or "s3" in s:
            return "S3"
        if "cloudfront" in s:
            return "CloudFront"
        if "nat gateway" in s:
            return "NAT Gateway"
        if "elastic block store" in s or "ebs" in s:
            return "EBS"
        if "lambda" in s:
            return "Lambda"
        return service
