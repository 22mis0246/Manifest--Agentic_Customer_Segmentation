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
                },
                "group_by_segment": {
                    "type": "boolean",
                    "description": "Set true when comparing metrics across customer segments.",
                },
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
            "required": [],
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
            "required": [],
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

from tools.eda_tool import eda_tool
from tools.explainability_tool import explainability_tool
from tools.feature_engineering_tool import feature_engineering_tool
from tools.recommendation_tool import recommendation_tool
from tools.segmentation_tool import segmentation_tool


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "eda_tool": eda_tool,
    "feature_engineering_tool": feature_engineering_tool,
    "segmentation_tool": segmentation_tool,
    "explainability_tool": explainability_tool,
    "recommendation_tool": recommendation_tool,
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
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None

    def trace_as_text(self) -> str:
        return "\n".join(f"[{e.step}] {e.detail}" for e in self.trace)


SYSTEM_PROMPT = (
    "You are SegmentIQ, an analytics agent for retail banking. "
    "Plan the minimum set of tools needed to answer the user's question. "
    "Call tools one at a time. Typical flows:\n"
    "- Segmentation requests: feature_engineering_tool -> segmentation_tool\n"
    "- Explain segment rules: explainability_tool (segment_name when known)\n"
    "- Segment comparisons: segmentation_tool if needed -> eda_tool with group_by_segment=true\n"
    "- Conversion/up-sell: recommendation_tool with conversion_query=true\n"
    "Use rule_based segmentation with 3 segments for priority/regular/dormant wording. "
    "If the request is ambiguous, ask a clarifying question instead of calling tools."
)

SYNTHESIS_PROMPT = (
    "You are SegmentIQ reporting to a bank analytics lead. "
    "Turn the tool outputs into a concise, human-readable answer with bullet insights, "
    "plain numbers, and recommended next actions. Do not mention internal tool names."
)

MAX_TOOL_STEPS = 5


# ---------------------------------------------------------------------------
# 4. The agent loop itself
# ---------------------------------------------------------------------------

class SegmentIQAgent:
    def __init__(self, model: str = "gemini-3.6-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Add it to your .env file or "
                "export it in your shell before running."
            )
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _execute_tool(self, tool_name: str, tool_params: dict[str, Any]) -> Any:
        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            raise KeyError(f"No implementation registered for {tool_name}")
        return tool_fn(**tool_params)

    def _synthesize(self, user_query: str, tools_used: list[dict[str, Any]]) -> str:
        payload = json.dumps(tools_used, default=str)
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                f"User question:\n{user_query}\n\n"
                f"Analytics outputs:\n{payload}\n\n"
                "Write the final answer for the bank team."
            ),
            config=types.GenerateContentConfig(system_instruction=SYNTHESIS_PROMPT),
        )
        return response.text or "Analysis complete. Review the structured output for details."

    def run(self, user_query: str) -> AgentResult:
        trace: list[TraceEvent] = [TraceEvent("received_query", user_query)]
        tools_used: list[dict[str, Any]] = []
        contents: list[Any] = [user_query]

        for step in range(MAX_TOOL_STEPS):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=_to_gemini_tools(TOOL_DEFINITIONS),
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                ),
            )

            parts = response.candidates[0].content.parts
            function_call = next((p.function_call for p in parts if p.function_call), None)
            text_part = next((p.text for p in parts if p.text), None)

            if function_call is None:
                if not tools_used:
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
                break

            tool_name = function_call.name
            tool_params = dict(function_call.args) if function_call.args else {}
            trace.append(
                TraceEvent(
                    "tool_selected",
                    f"step {step + 1}: {tool_name} {json.dumps(tool_params, default=str)}",
                )
            )

            try:
                result = self._execute_tool(tool_name, tool_params)
            except Exception as exc:
                trace.append(TraceEvent("error", str(exc)))
                return AgentResult(
                    query=user_query,
                    tool_called=tool_name,
                    tool_params=tool_params,
                    tool_output={"error": str(exc)},
                    trace=trace,
                    tools_used=tools_used,
                )

            tools_used.append({"tool": tool_name, "params": tool_params, "result": result})
            trace.append(TraceEvent("tool_executed", f"{tool_name} finished"))

            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": result},
                            )
                        )
                    ],
                )
            )

        summary = self._synthesize(user_query, tools_used)
        trace.append(TraceEvent("summary_ready", "Generated human-readable response"))

        last = tools_used[-1] if tools_used else {}
        return AgentResult(
            query=user_query,
            tool_called=last.get("tool"),
            tool_params=last.get("params", {}),
            tool_output=last.get("result"),
            trace=trace,
            tools_used=tools_used,
            summary=summary,
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