from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

CORE = [
    '/api/health',
    '/api/system/capabilities',
    '/api/billing/summary',
    '/api/metrics/overview',
    '/api/pricing/optimization-plan',
    '/api/ml/anomalies',
    '/api/ml/forecast?months=3',
    '/api/ml/service-analysis',
    '/api/rag/status',
    '/api/rag/documents',
    '/api/aws/status',
    '/api/aws/resource-overview',
    '/api/analytics/history',
    '/api/analytics/root-cause',
    '/api/evaluation/cases',
]

def test_core_get_endpoints():
    for path in CORE:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)


def test_empty_agent_request_is_rejected():
    response = client.post('/api/agent/run', json={'question': ''})
    assert response.status_code == 400


def test_aws_account_validation():
    response = client.post('/api/aws/test-connection', json={
        'account_id': '123',
        'access_key_id': 'x',
        'secret_access_key': 'y',
        'region': 'ap-south-1',
    })
    assert response.status_code == 400
