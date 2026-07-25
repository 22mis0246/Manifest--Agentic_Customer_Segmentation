"""Run all four challenge example queries through the live Gemini agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core import SegmentIQAgent
from data_store import store

QUERIES = [
    "Segment customers into priority, regular and dormant based on balance and transaction frequency",
    "On what basis were priority customers selected?",
    "What is the average transaction size for priority and regular customers?",
    "Which regular customers can be converted to priority customers? What should be done for the same?",
]


def main() -> None:
    store.auto_load()
    agent = SegmentIQAgent()

    for idx, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 70)
        print(f"Query {idx}: {query}")
        print("=" * 70)
        result = agent.run(query)
        print(result.trace_as_text())
        if result.needs_clarification:
            print(f"\nClarification: {result.clarifying_question}")
        else:
            print(f"\nTools used: {[t['tool'] for t in result.tools_used]}")
            print(f"\nSummary:\n{result.summary}")


if __name__ == "__main__":
    main()