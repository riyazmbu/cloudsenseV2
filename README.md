# CloudSense AI — Step 14: Semantic RAG + Evidence-Grounded Agent

Adds semantic RAG to Step 13 while preserving a deterministic fallback.

## What changed
- OpenAI `text-embedding-3-small` retrieval when `OPENAI_API_KEY` is configured.
- Local cached document embeddings in `knowledge_base/.embedding_cache.json`.
- Lexical retrieval fallback when embeddings are unavailable.
- RAG results include score, retrieval method, and evidence IDs such as `[rightsizing_policy.txt#0]`.
- Agent policy tool now returns evidence IDs for grounded citations.
- New `GET /api/rag/status` endpoint reports RAG mode and index size.
- Existing `/api/rag/search`, `/api/rag/documents`, AWS tools, ML tools, and Step 13 tool-calling agent remain intact.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
uvicorn app.main:app --reload
```

If `OPENAI_API_KEY` is omitted, the system still works using lexical retrieval.

## Product validation question
`Which production EC2 resources can I right-size, and what approval is required?`

The agent should combine AWS resource evidence with policy evidence and cite the returned policy chunk IDs.

## AI / ML architecture
- GenAI: Groq LLM (`CLOUDSENSE_AI_MODEL`) for grounded response synthesis and read-only tool-calling agent.
- RAG: ChromaDB persistent vector database + Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- ML: scikit-learn Isolation Forest for anomaly detection and K-Means for utilization/cost segmentation.
- Deterministic engine: service-specific optimization rules and savings estimates remain separate from ML/LLM.

## Local AWS authentication (current CloudSense approach)

CloudSense local development uses:

`CloudSenseConnector IAM User → STS AssumeRole → CloudSenseReadOnlyRole → temporary credentials`

Configure the IAM user's credentials on the backend machine only:

```text
CLOUDSENSE_AWS_ACCESS_KEY_ID=<CloudSenseConnector access key>
CLOUDSENSE_AWS_SECRET_ACCESS_KEY=<CloudSenseConnector secret key>
CLOUDSENSE_ROLE_SESSION_NAME=CloudSenseLocal
CLOUDSENSE_REQUIRE_STS=true
AWS_REGION=ap-south-1
```

The React UI does not accept AWS access keys, secret keys, organization SSO details, or External ID. It asks only for AWS Account ID, Region, and CloudSenseReadOnlyRole ARN. The backend calls STS AssumeRole and uses the returned temporary credentials for AWS reads.

See `../PRODUCTION_AWS_STS_SETUP.md` and `../docs/CloudSenseReadOnlyPolicy.json` for the current local setup.
