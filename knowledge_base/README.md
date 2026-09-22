# CloudSense AI — AWS Knowledge Base

This knowledge base is intentionally service-specific. It is guidance for explanation and validation; it does not override live AWS resource data.

Sources used for the current curated content: official AWS documentation / AWS Prescriptive Guidance for EC2 Compute Optimizer, RDS CloudWatch/Database Insights, EBS lifecycle and unattached-volume cleanup, NAT Gateway cost checks, S3 cost optimization/lifecycle, Lambda memory tuning, CloudFront cache optimization, and AWS Cost Explorer resource-level data.

Design rule:
- Live AWS data determines whether a specific resource is a candidate.
- This knowledge base explains why the recommendation is appropriate and what must be validated.
- Never infer a resource-level saving when resource-level cost is unavailable.
- Never apply EC2 CPU/memory rightsizing rules to NAT Gateway, S3, Lambda, CloudFront, or EBS.
