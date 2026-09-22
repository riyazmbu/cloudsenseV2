from __future__ import annotations

from typing import List, Dict, Any
import math
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _features(records):
    rows = []
    refs = []
    for r in records:
        rows.append([
            float(r.monthly_cost or 0),
            float(r.cpu_utilization or 0),
            float(r.memory_utilization or 0),
            float(r.hours_running or 0),
        ])
        refs.append(r)
    return np.asarray(rows, dtype=float), refs


def detect_cost_anomalies(records) -> List[Dict[str, Any]]:
    """Unsupervised ML anomaly detection using Isolation Forest.

    Features: monthly cost, CPU, memory and running hours. The model learns the
    distribution of the uploaded dataset rather than using a hard-coded cost threshold.
    """
    X, refs = _features(records)
    if len(refs) < 5:
        raise ValueError("IsolationForest requires at least 5 resource records")

    Xs = StandardScaler().fit_transform(X)
    contamination = min(0.10, max(0.02, 5 / len(refs)))
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    labels = model.fit_predict(Xs)
    scores = -model.decision_function(Xs)
    scale = max(float(np.max(scores)), 1e-9)

    results = []
    for r, label, score in zip(refs, labels, scores):
        normalized = min(1.0, max(0.0, float(score / scale)))
        results.append({
            "resource_id": r.resource_id,
            "service": r.service,
            "environment": r.environment,
            "monthly_cost": round(float(r.monthly_cost or 0), 2),
            "ml_anomaly_score": round(normalized, 4),
            "anomaly": bool(label == -1),
            "severity": "high" if normalized >= 0.75 else "medium" if normalized >= 0.5 else "normal",
            "ml_method": "IsolationForest",
        })
    return results


def utilization_clusters(records, n_clusters: int = 3):
    """K-Means segmentation of resources by utilization/cost profile."""
    X, refs = _features(records)
    if len(refs) < 3:
        raise ValueError("KMeans requires at least 3 resource records")
    k = min(max(2, n_clusters), len(refs))
    Xs = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)
    clusters = []
    for idx, (r, label) in enumerate(zip(refs, labels)):
        clusters.append({
            "resource_id": r.resource_id,
            "service": r.service,
            "cluster": int(label),
            "cpu_utilization": float(r.cpu_utilization or 0),
            "memory_utilization": float(r.memory_utilization or 0),
            "monthly_cost": float(r.monthly_cost or 0),
        })
    return {"method": "KMeans", "cluster_count": k, "clusters": clusters, "model_used": True, "features": ["monthly_cost", "cpu_utilization", "memory_utilization", "hours_running"]}


def service_cost_anomalies(records):
    service_values = {}
    for r in records:
        key = r.service or "Unknown"
        service_values.setdefault(key, []).append(float(r.monthly_cost or 0))
    output = []
    for service, values in service_values.items():
        avg = sum(values) / len(values) if values else 0
        output.append({
            "service": service,
            "resource_count": len(values),
            "total_monthly_cost": round(sum(values), 2),
            "average_resource_cost": round(avg, 2),
            "max_resource_cost": round(max(values), 2) if values else 0,
        })
    return sorted(output, key=lambda x: x["total_monthly_cost"], reverse=True)


def forecast_monthly_cost(records, months=3):
    current = sum(float(r.monthly_cost or 0) for r in records)
    return {
        "current_monthly_cost": round(current, 2),
        "forecast": [{"month_offset": i, "forecast_monthly_cost": round(current, 2), "method": "current_run_rate_baseline"} for i in range(1, months + 1)],
        "note": "Forecast is a baseline because a single billing period does not provide enough history for supervised time-series training.",
    }


def ml_status():
    return {
        "available": True,
        "models": [
            {"name": "IsolationForest", "purpose": "resource/cost anomaly detection"},
            {"name": "KMeans", "purpose": "resource utilization/cost clustering"},
        ],
        "framework": "scikit-learn",
        "training_mode": "dataset-fit unsupervised learning",
        "features": ["monthly_cost", "cpu_utilization", "memory_utilization", "hours_running"],
    }
