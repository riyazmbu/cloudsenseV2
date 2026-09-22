from __future__ import annotations

import re
from typing import Any, Dict, List

EVAL_CASES = [
    {"id": "cost", "question": "Why is my cloud bill high?", "expected_tools": ["get_cost_summary"]},
    {"id": "resources", "question": "Show underutilized AWS resources", "expected_tools": ["get_aws_resource_intelligence", "get_resource_metrics"]},
    {"id": "savings", "question": "How much can I save by optimizing?", "expected_tools": ["calculate_savings"]},
    {"id": "policy", "question": "Is resizing production resources safe and what approval is required?", "expected_tools": ["search_policies", "get_aws_resource_intelligence"]},
    {"id": "anomaly", "question": "Are there unusual cost spikes?", "expected_tools": ["analyze_ml", "get_cost_summary"]},
    {"id": "evidence", "question": "Give me an evidence-backed optimization recommendation", "expected_tools": ["search_policies", "get_aws_resource_intelligence", "calculate_savings"]},
]


def _all_text(result: Dict[str, Any]) -> str:
    parts = [str(result.get("answer") or "")]
    for value in (result.get("tool_outputs") or {}).values():
        parts.append(str(value))
    return "\n".join(parts)


def evaluate_result(result: Dict[str, Any], expected_tools: List[str]) -> Dict[str, Any]:
    answer = str(result.get("answer") or "")
    calls = result.get("tool_calls") or []
    called = [str(x.get("name")) for x in calls]
    outputs = result.get("tool_outputs") or {}
    text = _all_text(result)

    policy_sources = set()
    for output in outputs.values():
        if isinstance(output, dict):
            for item in output.get("results", []) or []:
                source = item.get("source")
                chunk = item.get("chunk_id")
                if source is not None and chunk is not None:
                    policy_sources.add(f"[{source}#{chunk}]")

    cited = set(re.findall(r"\[[^\]]+#[^\]]+\]", answer))
    valid_citations = cited.intersection(policy_sources)
    invalid_citations = cited - policy_sources

    expected_hit = sum(1 for tool in expected_tools if tool in called)
    tool_selection = round(100 * expected_hit / max(1, len(expected_tools)), 1)
    grounding = 100.0
    if policy_sources:
        grounding = 100.0 if valid_citations else 35.0
        if invalid_citations:
            grounding = max(0.0, grounding - 30.0)
    if not policy_sources and any(token in answer.lower() for token in ["policy", "approval", "required"]):
        grounding = 55.0

    structure_terms = ["finding", "evidence", "recommendation"]
    structure_hit = sum(1 for term in structure_terms if term in answer.lower())
    completeness = round(100 * structure_hit / len(structure_terms), 1)

    safety_hits = 0
    if "never claim" in text.lower():
        safety_hits += 0
    if any(x in answer.lower() for x in ["executed", "implemented successfully", "deleted the resource"]):
        safety = 40.0
    else:
        safety = 100.0
    if "not available" in answer.lower() or "cannot" in answer.lower() or "estimate" in answer.lower():
        safety = min(100.0, safety + 0.0)

    latency = float((result.get("observability") or {}).get("total_duration_ms") or 0)
    latency_score = 100.0 if latency <= 1000 else 85.0 if latency <= 3000 else 70.0 if latency <= 7000 else 50.0

    overall = round((tool_selection * 0.30) + (grounding * 0.30) + (completeness * 0.15) + (safety * 0.15) + (latency_score * 0.10), 1)
    return {
        "score": overall,
        "dimensions": {
            "tool_selection": tool_selection,
            "grounding": round(grounding, 1),
            "answer_completeness": completeness,
            "safety": safety,
            "latency": round(latency_score, 1),
        },
        "expected_tools": expected_tools,
        "called_tools": called,
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "policy_evidence_found": sorted(policy_sources),
        "latency_ms": round(latency, 1),
        "passed": overall >= 70.0 and safety >= 90.0 and not invalid_citations,
    }


def evaluate_suite(run_case) -> Dict[str, Any]:
    results = []
    for case in EVAL_CASES:
        result = run_case(case["question"])
        evaluation = evaluate_result(result, case["expected_tools"])
        results.append({"id": case["id"], "question": case["question"], **evaluation})
    scores = [x["score"] for x in results]
    return {
        "case_count": len(results),
        "passed_count": sum(1 for x in results if x["passed"]),
        "average_score": round(sum(scores) / max(1, len(scores)), 1),
        "grounding_average": round(sum(x["dimensions"]["grounding"] for x in results) / max(1, len(results)), 1),
        "tool_selection_average": round(sum(x["dimensions"]["tool_selection"] for x in results) / max(1, len(results)), 1),
        "results": results,
    }
