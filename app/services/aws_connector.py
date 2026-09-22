import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

CONNECTION_FILE = Path(__file__).resolve().parents[2] / "aws_connection.json"

# Runtime-only configuration. Permanent customer credentials are never persisted.
ROLE_SESSION_NAME = os.getenv("CLOUDSENSE_ROLE_SESSION_NAME", "CloudSenseReadOnly")
SOURCE_IDENTITY = os.getenv("CLOUDSENSE_SOURCE_IDENTITY")
REQUIRE_STS = os.getenv("CLOUDSENSE_REQUIRE_STS", "true").lower() in {"1", "true", "yes", "on"}

_RUNTIME_SESSION = None
_RUNTIME_EXPIRATION = None


def _save_state(state: dict):
    # Never persist secrets, ExternalId, or temporary credentials.
    safe = dict(state)
    safe.pop("external_id", None)
    safe.pop("access_key_id", None)
    safe.pop("secret_access_key", None)
    CONNECTION_FILE.write_text(json.dumps(safe, indent=2), encoding="utf-8")


def get_connection_state():
    if not CONNECTION_FILE.exists():
        return {"connected": False}
    try:
        return json.loads(CONNECTION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"connected": False}


def _base_session(access_key_id: str, secret_access_key: str, region: str):
    return boto3.Session(
        aws_access_key_id=access_key_id.strip(),
        aws_secret_access_key=secret_access_key.strip(),
        region_name=region.strip(),
    )


def _bootstrap_session(region: str):
    """Create the local CloudSense bootstrap identity.

    Local development uses the dedicated IAM user CloudSenseConnector.
    Its long-lived access key is used only to call STS AssumeRole.
    The returned temporary credentials are then used for all AWS reads.

    Credentials are intentionally kept server-side and are never accepted
    from the React/browser request. The backend can also fall back to the
    standard boto3 credential chain for controlled development environments.
    """
    access_key = os.getenv("CLOUDSENSE_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("CLOUDSENSE_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

    if access_key and secret_key:
        return _base_session(access_key, secret_key, region)

    profile = os.getenv("CLOUDSENSE_AWS_PROFILE") or os.getenv("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region.strip())
    else:
        session = boto3.Session(region_name=region.strip())

    credentials = session.get_credentials()
    if credentials is None:
        raise NoCredentialsError(error_msg=(
            "CloudSense has no local IAM bootstrap credentials. Set "
            "CLOUDSENSE_AWS_ACCESS_KEY_ID and CLOUDSENSE_AWS_SECRET_ACCESS_KEY "
            "for the CloudSenseConnector IAM user, or configure an AWS profile. "
            "The bootstrap identity needs sts:AssumeRole on CloudSenseReadOnlyRole."
        ))
    return session


def bootstrap_identity(region: str):
    """Return non-secret information about the CloudSense caller identity."""
    session = _bootstrap_session(region)
    identity = session.client("sts", region_name=region.strip()).get_caller_identity()
    source = "CLOUDSENSE_AWS_ACCESS_KEY_ID / CLOUDSENSE_AWS_SECRET_ACCESS_KEY" if (os.getenv("CLOUDSENSE_AWS_ACCESS_KEY_ID") and os.getenv("CLOUDSENSE_AWS_SECRET_ACCESS_KEY")) else (os.getenv("CLOUDSENSE_AWS_PROFILE") or os.getenv("AWS_PROFILE") or "AWS SDK default credential chain")
    return identity, source


def validate_target_role(account_id: str, role_arn: str):
    if not role_arn or not role_arn.strip().startswith("arn:aws:iam::") or ":role/" not in role_arn:
        raise ValueError("Role ARN must look like arn:aws:iam::<account-id>:role/<role-name>.")
    role_account = role_arn.split(":")[4]
    if account_id and role_account != account_id.strip():
        raise ValueError(f"Role ARN account {role_account} does not match AWS Account ID {account_id.strip()}.")
    return role_account


def assume_role_session(account_id: str, role_arn: str, external_id: str, region: str):
    """Assume the customer's read-only role using the CloudSense runtime identity."""
    validate_target_role(account_id, role_arn)
    bootstrap = _bootstrap_session(region)
    sts = bootstrap.client("sts", region_name=region.strip())

    session_name = ROLE_SESSION_NAME.strip() or "CloudSenseReadOnly"
    if not all(c.isalnum() or c in "_+=,.@-" for c in session_name) or not 2 <= len(session_name) <= 64:
        raise ValueError("CLOUDSENSE_ROLE_SESSION_NAME must be 2-64 characters and contain only letters, numbers, _+=,.@-." )

    params = {
        "RoleArn": role_arn.strip(),
        "RoleSessionName": session_name,
        "DurationSeconds": 3600,
    }
    if external_id and external_id.strip():
        params["ExternalId"] = external_id.strip()
    if SOURCE_IDENTITY:
        source = SOURCE_IDENTITY.strip()
        if not 2 <= len(source) <= 64 or source.startswith("aws:") or not all(c.isalnum() or c in "_+=,.@-" for c in source):
            raise ValueError("CLOUDSENSE_SOURCE_IDENTITY must be 2-64 characters, cannot start with aws:, and must use valid STS source-identity characters.")
        params["SourceIdentity"] = source

    assumed = sts.assume_role(**params)
    creds = assumed["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=region.strip(),
    ), assumed


def _run_checks(session, region: str):
    sts = session.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    checks = {"identity": True, "billing": False, "cloudwatch": False, "ec2": False, "cloudtrail": False}
    errors = {}

    try:
        ce = session.client("ce", region_name="us-east-1")
        today = datetime.now(timezone.utc)
        ce.get_cost_and_usage(
            TimePeriod={"Start": (today - timedelta(days=2)).strftime("%Y-%m-%d"), "End": today.strftime("%Y-%m-%d")},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
        )
        checks["billing"] = True
    except Exception as exc:
        errors["billing"] = str(exc)

    try:
        session.client("cloudwatch", region_name=region).list_metrics(RecentlyActive="PT3H")
        checks["cloudwatch"] = True
    except Exception as exc:
        errors["cloudwatch"] = str(exc)

    try:
        session.client("ec2", region_name=region).describe_instances(MaxResults=5)
        checks["ec2"] = True
    except Exception as exc:
        errors["ec2"] = str(exc)

    try:
        session.client("cloudtrail", region_name=region).lookup_events(MaxResults=1)
        checks["cloudtrail"] = True
    except Exception as exc:
        errors["cloudtrail"] = str(exc)

    return identity, checks, errors


def test_connection(account_id: str, region: str, role_arn: str | None = None,
                    external_id: str | None = None, access_key_id: str | None = None,
                    secret_access_key: str | None = None):
    """Validate the target AWS account using STS AssumeRole.

    Permanent access-key authentication is intentionally disabled when
    CLOUDSENSE_REQUIRE_STS=true (the production default).
    """
    if not region or not region.strip():
        return {"success": False, "message": "AWS Region is required."}
    account_id = (account_id or "").strip()
    region = region.strip()

    global _RUNTIME_SESSION, _RUNTIME_EXPIRATION
    try:
        if REQUIRE_STS:
            if not role_arn or not role_arn.strip():
                return {"success": False, "message": "Role ARN is required. CloudSense production mode uses STS AssumeRole and temporary credentials."}
            session, assumed = assume_role_session(account_id, role_arn, external_id or "", region)
            expiration = assumed.get("Credentials", {}).get("Expiration")
            auth_mode = "assume_role"
        else:
            if role_arn and role_arn.strip():
                session, assumed = assume_role_session(account_id, role_arn, external_id or "", region)
                expiration = assumed.get("Credentials", {}).get("Expiration")
                auth_mode = "assume_role"
            elif access_key_id and secret_access_key:
                session = _base_session(access_key_id, secret_access_key, region)
                expiration = None
                auth_mode = "access_key_legacy"
            else:
                return {"success": False, "message": "Role ARN is required, or disable CLOUDSENSE_REQUIRE_STS only for controlled development testing."}

        _RUNTIME_SESSION = session
        _RUNTIME_EXPIRATION = expiration

        identity, checks, errors = _run_checks(session, region)
        actual_account = identity.get("Account")
        if account_id and actual_account != account_id:
            return {"success": False, "message": f"AWS Account ID mismatch. Assumed credentials belong to account {actual_account}, not {account_id}.", "account_id": actual_account}

        state = {
            "connected": bool(checks["identity"] and checks["billing"] and checks["ec2"]),
            "account_id": actual_account,
            "region": region,
            "auth_mode": auth_mode,
            "role_arn": role_arn if auth_mode == "assume_role" else None,
            "temporary_credentials": auth_mode == "assume_role",
            "credentials_expire_at": expiration.isoformat() if hasattr(expiration, "isoformat") else expiration,
            "last_tested_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "errors": errors,
            "caller_arn": identity.get("Arn"),
        }
        _save_state(state)

        if state["connected"]:
            message = "AWS connected successfully using STS AssumeRole." if auth_mode == "assume_role" else "AWS connected using legacy credentials."
        else:
            failed = [name.title() for name in ("identity", "billing", "ec2") if not checks.get(name)]
            message = "AWS connection failed. Required checks failed: " + ", ".join(failed) + "." if failed else "AWS connection failed."
        return {"success": state["connected"], "message": message, **state}
    except (ClientError, BotoCoreError, NoCredentialsError, ValueError) as exc:
        return {"success": False, "message": f"AWS connection failed: {exc}"}
    except Exception as exc:
        return {"success": False, "message": f"AWS connection failed: {exc}"}


def create_session_from_state(region: str | None = None):
    """Return the in-memory temporary session, refreshing it with STS when needed."""
    global _RUNTIME_SESSION, _RUNTIME_EXPIRATION
    if _RUNTIME_SESSION is not None:
        if _RUNTIME_EXPIRATION is None or _RUNTIME_EXPIRATION > datetime.now(timezone.utc) + timedelta(minutes=2):
            return _RUNTIME_SESSION

    state = get_connection_state()
    role_arn = state.get("role_arn")
    if role_arn:
        # ExternalId is intentionally not persisted. The deployment should provide
        # the same ExternalId through a secure server-side environment variable.
        external_id = os.getenv("CLOUDSENSE_DEFAULT_EXTERNAL_ID", "")
        session, assumed = assume_role_session(
            state.get("account_id", ""),
            role_arn,
            external_id,
            region or state.get("region") or "ap-south-1",
        )
        _RUNTIME_SESSION = session
        _RUNTIME_EXPIRATION = assumed.get("Credentials", {}).get("Expiration")
        return session

    raise RuntimeError("AWS session is unavailable or expired. Test the STS connection again.")
