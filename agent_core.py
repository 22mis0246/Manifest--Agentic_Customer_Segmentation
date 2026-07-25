"""
SegmentIQ — Agent Core (Gemini 2.5 Flash only)
=================================================
This is the orchestration brain of the agent. It does NOT contain any
banking/ML logic itself — it only:

  1. Receives a natural-language query from the user
  2. Asks Gemini 2.5 Flash which tool(s) to call and with what parameters
  3. Executes the corresponding Python function
  4. Logs every decision to a visible reasoning trace
  5. Returns the raw result back to the caller (Streamlit app, API, etc.)

Real tool implementations (eda_tool, feature_engineering_tool, etc.) are
imported from separate modules and plugged in via TOOL_REGISTRY. Until
those modules exist, each tool safely falls back to a placeholder so the
agent loop can be tested end-to-end from minute one.

Environment variable expected (in a .env file or the shell env):
    GEMINI_API_KEY   — your Gemini API key
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory into os.environ

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# 1. Tool contracts — the schema the LLM uses to decide what to call
# ---------------------------------------------------------------------------
# Each entry mirrors Gemini's function-declaration spec. Keep descriptions
# specific — vague descriptions are the #1 reason function-calling picks
# the wrong tool.

TOOL_DEFINITIONS = [
    {
        "name": "eda_tool",
        "description": (
            "Run exploratory data analysis on the customer dataset: "
            "missing values, distributions, correlations, and outlier flags. "
            "Use when the user asks about data quality, distributions, "
            "correlations, or wants a general understanding of the dataset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to analyze. Empty list = analyze all columns.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "feature_engineering_tool",
        "description": (
            "Derive behavioural/financial features from raw transaction data "
            "(e.g. average monthly balance, transaction frequency, average "
            "transaction size, recency). Use when the user's request implies "
            "features need to be built before segmentation or analysis can happen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "features_requested": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which derived features to compute, e.g. ['avg_monthly_balance', 'tx_frequency'].",
                }
            },
            "required": ["features_requested"],
        },
    },
    {
        "name": "segmentation_tool",
        "description": (
            "Group customers into segments based on given criteria. Use when "
            "the user asks to segment/cluster/group customers, or asks how "
            "many segments exist and their sizes. This tool is generic — it "
            "accepts ANY set of features, not just balance/frequency."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature names to segment on, e.g. ['balance', 'tx_frequency'].",
                },
                "num_segments": {
                    "type": "integer",
                    "description": "Number of segments to create. Default 3 if not specified.",
                },
                "method": {
                    "type": "string",
                    "enum": ["rule_based", "kmeans"],
                    "description": "Segmentation approach. Default rule_based unless user asks for ML/clustering explicitly.",
                },
            },
            "required": ["criteria"],
        },
    },
    {
        "name": "explainability_tool",
        "description": (
            "Explain WHY a specific customer (or segment) was assigned to a "
            "segment — returns the rule thresholds crossed or the feature "
            "importances driving the assignment. Use when the user asks "
            "'why', 'on what basis', or 'what distinguishes' a segment/customer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Specific customer to explain. Omit if explaining a segment in general.",
                },
                "segment_name": {
                    "type": "string",
                    "description": "Segment to explain, e.g. 'priority'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "recommendation_tool",
        "description": (
            "Recommend products/actions for a segment, or identify customers "
            "close to crossing into a higher-value segment along with what "
            "would move them there. Use for cross-sell/up-sell or conversion "
            "questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_segment": {
                    "type": "string",
                    "description": "The segment to recommend for, e.g. 'priority', 'regular'.",
                },
                "conversion_query": {
                    "type": "boolean",
                    "description": "True if the user is asking how to move customers INTO a higher segment.",
                },
            },
            "required": [],
        },
    },
]


def _to_gemini_tools(defs: list[dict]):
    """Wrap our neutral schema into Gemini's Tool/FunctionDeclaration objects."""
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=d["name"],
                    description=d["description"],
                    parameters=d["parameters"],
                )
                for d in defs
            ]
        )
    ]


# ---------------------------------------------------------------------------
# 2. Placeholder tool bodies — replaced by real imports as each module lands
# ---------------------------------------------------------------------------

def _not_implemented(tool_name: str, **kwargs) -> dict:
    return {
        "status": "not_implemented",
        "tool": tool_name,
        "received_params": kwargs,
        "note": f"{tool_name} is wired into the agent but has no logic yet.",
    }


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "eda_tool": lambda **kw: _not_implemented("eda_tool", **kw),
    "feature_engineering_tool": lambda **kw: _not_implemented("feature_engineering_tool", **kw),
    "segmentation_tool": lambda **kw: _not_implemented("segmentation_tool", **kw),
    "explainability_tool": lambda **kw: _not_implemented("explainability_tool", **kw),
    "recommendation_tool": lambda **kw: _not_implemented("recommendation_tool", **kw),
}


def register_tool(name: str, fn: Callable[..., Any]) -> None:
    """Swap a placeholder for the real implementation once a teammate finishes it.

    Example:
        from eda import eda_tool
        register_tool("eda_tool", eda_tool)
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool '{name}'. Add it to TOOL_DEFINITIONS first.")
    TOOL_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# 3. Reasoning trace — visible log of every decision the agent makes
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    step: str
    detail: str


@dataclass
class AgentResult:
    query: str
    tool_called: str | None
    tool_params: dict
    tool_output: Any
    needs_clarification: bool = False
    clarifying_question: str | None = None
    trace: list[TraceEvent] = field(default_factory=list)

    def trace_as_text(self) -> str:
        return "\n".join(f"[{e.step}] {e.detail}" for e in self.trace)


SYSTEM_PROMPT = (
    "You are the routing brain for SegmentIQ, a retail banking "
    "customer segmentation agent. Given a user's natural language "
    "question, decide which single tool best answers it and extract "
    "the parameters from the query. If the query is ambiguous or "
    "missing information needed to call a tool correctly (e.g. no "
    "criteria given for segmentation), do NOT guess — ask a "
    "clarifying question in plain text instead of calling a tool."
)


# ---------------------------------------------------------------------------
# 4. The agent loop itself
# ---------------------------------------------------------------------------

class SegmentIQAgent:
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Add it to your .env file or "
                "export it in your shell before running."
            )
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def run(self, user_query: str) -> AgentResult:
        trace: list[TraceEvent] = [TraceEvent("received_query", user_query)]

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=_to_gemini_tools(TOOL_DEFINITIONS),
                # AUTO lets the model choose freely between calling a tool
                # or replying in plain text (needed for clarifying questions).
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            ),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_call = next((p.function_call for p in parts if p.function_call), None)
        text_part = next((p.text for p in parts if p.text), None)

        # Case 1: Gemini asked for clarification instead of calling a tool
        if function_call is None:
            question = text_part or "Could you clarify your request?"
            trace.append(TraceEvent("clarification_needed", question))
            return AgentResult(
                query=user_query,
                tool_called=None,
                tool_params={},
                tool_output=None,
                needs_clarification=True,
                clarifying_question=question,
                trace=trace,
            )

        # Case 2: Gemini selected a tool — execute it
        tool_name = function_call.name
        tool_params = dict(function_call.args) if function_call.args else {}
        trace.append(
            TraceEvent(
                "tool_selected",
                f"{tool_name} with params {json.dumps(tool_params, default=str)}",
            )
        )

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            trace.append(TraceEvent("error", f"No implementation registered for {tool_name}"))
            return AgentResult(user_query, tool_name, tool_params, None, trace=trace)

        result = tool_fn(**tool_params)
        trace.append(TraceEvent("tool_executed", f"{tool_name} returned a result"))

        return AgentResult(
            query=user_query,
            tool_called=tool_name,
            tool_params=tool_params,
            tool_output=result,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# 5. Quick manual test — run this file directly to sanity-check routing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = SegmentIQAgent()

    test_queries = [
        "Segment customers into priority, regular and dormant based on balance and transaction frequency",
        "On what basis were priority customers selected?",
        "What is the average transaction size for priority and regular customers?",
        "Which regular customers can be converted to priority customers?",
    ]

    for q in test_queries:
        print("=" * 70)
        result = agent.run(q)
        print(f"QUERY: {q}")
        print(result.trace_as_text())
        if result.needs_clarification:
            print(f"CLARIFYING QUESTION: {result.clarifying_question}")
        else:
            print(f"TOOL: {result.tool_called} | PARAMS: {result.tool_params}")
            print(f"OUTPUT: {result.tool_output}")