from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

AGENT_SYSTEM_PROMPT = """You are CloudSense AI, an enterprise cloud cost intelligence agent.
Use the read-only tools provided to gather evidence before answering. Do not invent costs,
resources, metrics, savings or policy requirements. For recommendations, distinguish observed
data from estimates. Use policy/RAG evidence when supplied. Never claim an AWS change was executed.
Answer in simple business language with: Finding, Evidence, Recommendation, Savings/Impact, Validation.
"""

TOOLS = [
    {"type": "function", "function": {"name": "get_cost_summary", "description": "Get current CloudSense cost totals and resource count.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_resource_metrics", "description": "Get resource utilization and optimization candidate evidence.", "parameters": {"type": "object", "properties": {"resource_id": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "calculate_savings", "description": "Calculate deterministic CloudSense savings from all optimization candidates.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search_policies", "description": "Search the CloudSense RAG knowledge base for AWS policy and service guidance.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "analyze_ml", "description": "Run ML anomaly detection and utilization clustering over current billing/resource data.", "parameters": {"type": "object", "properties": {}}}},
]


class CloudSenseAgent:
    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self.tools = tools
        self.model = os.getenv("CLOUDSENSE_AI_MODEL", "openai/gpt-oss-120b")

    def run(self, question: str) -> Dict[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        if Groq is None:
            raise RuntimeError("groq package is not installed")

        client = Groq(api_key=api_key)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        used_tools = []

        for _ in range(int(os.getenv("CLOUDSENSE_AGENT_MAX_ROUNDS", "4"))):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=900,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return {"answer": msg.content or "No answer was produced.", "agent_mode": "groq_tool_calling", "model": self.model, "tool_calls": used_tools}

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls]})
            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name not in self.tools:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    if name == "search_policies":
                        result = self.tools[name](args.get("question", question))
                    elif name == "get_resource_metrics":
                        result = self.tools[name](args.get("resource_id")) if args.get("resource_id") else self.tools[name]()
                    else:
                        result = self.tools[name]()
                used_tools.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)[:12000]})

        raise RuntimeError("GenAI agent reached its maximum tool-call rounds without producing a grounded answer")
