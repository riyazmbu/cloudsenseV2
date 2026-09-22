import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.ml_engine import detect_cost_anomalies, utilization_clusters
from app.services.rag import load_documents, retrieve, HF_EMBEDDING_MODEL


def make_records():
    return [SimpleNamespace(
        resource_id=f"i-{i:04d}", service="EC2", environment="prod",
        monthly_cost=100 + i * 10, cpu_utilization=20 + i,
        memory_utilization=25 + i, hours_running=700,
    ) for i in range(12)]


def test_ml_isolation_forest_output_shape():
    result = detect_cost_anomalies(make_records())
    assert len(result) == 12
    assert all("ml_anomaly_score" in x for x in result)
    assert all(x["ml_method"] == "IsolationForest" for x in result)


def test_ml_kmeans_output():
    result = utilization_clusters(make_records())
    assert result["method"] == "KMeans"
    assert result["model_used"] is True
    assert len(result["clusters"]) == 12


def test_rag_knowledge_exists():
    docs = load_documents()
    names = {x["source"] for x in docs}
    assert "cloudsense_rag_methodology.txt" in names
    assert "cloudsense_genai_guardrails.txt" in names
    assert "cloudsense_ml_methodology.txt" in names
    assert HF_EMBEDDING_MODEL.startswith("sentence-transformers/")


def test_rag_returns_semantic_vector_evidence():
    results = retrieve("How should I optimize an EC2 right-sizing candidate?")
    assert results
    assert all(x["retrieval"] == "huggingface_embedding+chromadb" for x in results)
    assert any("ec2" in x["text"].lower() for x in results)
