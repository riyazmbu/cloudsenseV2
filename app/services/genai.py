import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from groq import Groq


SYSTEM_PROMPT = """
You are CloudSense AI, a cloud cost optimization copilot.

Explain CloudSense findings in simple business language. Never dump raw JSON or a long technical report.

Rules:
- Use only supplied CloudSense data and supplied RAG knowledge.
- Never invent resources, costs, metrics or savings.
- Use deterministic savings values supplied by CloudSense.
- For resource-specific questions, prioritize the supplied resource data.
- Use RAG knowledge to explain the AWS/service-specific reasoning.
- Do not claim that any AWS action has been executed.
- Recommendations require validation and human approval.

For a resource question, explain:

1. What is happening
2. Evidence
3. Why it matters
4. Recommended action
5. Estimated savings, if available
6. Validation required

Be concise and actionable. Use a clear structure with a short headline, key numbers, plain-English explanation, recommended next steps, and a brief validation note. Bold important values using Markdown **bold**. Prefer bullets over dense paragraphs.
"""


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_resource_id(question: str) -> Optional[str]:

    patterns = [
        r"\bi-[a-zA-Z0-9-]+\b",
        r"\bvol-[a-zA-Z0-9-]+\b",
        r"\bdb-[a-zA-Z0-9-]+\b",
        r"\bnat-[a-zA-Z0-9-]+\b",
        r"\bfn-[a-zA-Z0-9-]+\b",
        r"\bbucket-[a-zA-Z0-9-]+\b",
        r"\bdist-[a-zA-Z0-9-]+\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)

        if match:
            return match.group(0)

    return None


def find_resource(resource_id, records, pricing):

    target = resource_id.lower()

    for record in records:

        rid = str(
            record.get("resource_id") or ""
        ).lower()

        if rid == target:
            return record

    for candidate in pricing:

        rid = str(
            candidate.get("resource_id") or ""
        ).lower()

        if rid == target:
            return candidate

    return None


def build_resource_context(resource_id, records, pricing):

    if not resource_id:
        return None

    resource = find_resource(
        resource_id,
        records,
        pricing
    )

    if not resource:
        return None

    result = {
        "resource_id": resource.get("resource_id"),
        "service": resource.get("service"),
        "resource_name": resource.get("resource_name"),
        "region": resource.get("region"),
        "environment": resource.get("environment"),
        "team": resource.get("team"),
        "state": resource.get("state"),

        "cpu_utilization": safe_float(
            resource.get("cpu_utilization")
        ),

        "memory_utilization": safe_float(
            resource.get("memory_utilization")
        ),

        "monthly_cost": safe_float(
            resource.get("monthly_cost")
        ),

        "cost_currency": resource.get(
            "cost_currency"
        ),
    }

    # Find recommendation
    for candidate in pricing:

        rid = str(
            candidate.get("resource_id") or ""
        ).lower()

        if rid == resource_id.lower():

            result["optimization_candidate"] = (
                candidate.get(
                    "optimization_candidate"
                )
            )

            result["recommendation_type"] = (
                candidate.get(
                    "recommendation_type"
                )
            )

            result["estimated_monthly_saving"] = (
                safe_float(
                    candidate.get(
                        "estimated_monthly_saving"
                    )
                )
            )

            result["recommendation_reason"] = (
                candidate.get("reason")
                or candidate.get(
                    "recommendation_reason"
                )
            )

            result["evidence"] = (
                candidate.get("evidence")
            )

            break

    return result


# ---------------------------------------------------------
# General context
# ---------------------------------------------------------

def build_context(
    records,
    pricing,
    overview,
    ml_insights
):

    top = sorted(
        records,
        key=lambda x: safe_float(
            x.get("monthly_cost"),
            0
        ),
        reverse=True
    )[:10]

    candidates = [
        x for x in pricing
        if x.get("optimization_candidate")
    ]

    candidates = sorted(
        candidates,
        key=lambda x: safe_float(
            x.get(
                "estimated_monthly_saving"
            ),
            0
        ),
        reverse=True
    )[:10]

    return {
        "overview": overview,
        "ml_insights": ml_insights,
        "top_cost_resources": top,
        "optimization_candidates": candidates,
    }


# ---------------------------------------------------------
# Main answer function
# ---------------------------------------------------------

def answer_question(
    question: str,
    context: dict,
    rag_context: str,
    records=None,
    pricing=None,
    ml_evidence=None,
):
    """Strict GenAI pipeline. No fallback or synthetic AI response is allowed."""
    if not question.strip():
        raise ValueError("question cannot be empty")
    if not rag_context.strip():
        raise RuntimeError("RAG evidence is required for GenAI responses")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("CLOUDSENSE_AI_MODEL", "openai/gpt-oss-120b")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    if Groq is None:
        raise RuntimeError("groq package is not installed")

    resource_id = extract_resource_id(question)
    resource_context = build_resource_context(resource_id, records or [], pricing or [])
    if resource_id and not resource_context:
        raise ValueError(f"Resource {resource_id} was not found in CloudSense data")

    focused_context = {
        "cloudsense_data": context,
        "resource_requested": resource_id,
        "resource_data": resource_context,
        "ml_evidence": ml_evidence or {},
        "rag_evidence": rag_context,
    }

    user_prompt = f"""
CloudSense evidence package:
{focused_context}

Customer question:
{question}

Answer using ONLY the evidence package.
Do not invent metrics, costs, savings, resources, AWS actions, or policies.
Distinguish observed facts, ML findings, retrieved knowledge, and CloudSense estimates.
Cite retrieved knowledge inline using the supplied Evidence IDs when making AWS/service-policy claims.
If required evidence is missing, say exactly what evidence is missing instead of guessing.
"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=900,
    )

    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("GenAI provider returned an empty response")

    return {
        "answer": response.choices[0].message.content,
        "provider": "groq",
        "model": model,
        "used_llm": True,
        "resource_id": resource_id,
        "rag_used": True,
        "rag_retrieval": "chromadb+huggingface",
        "ml_used": bool(ml_evidence),
    }
