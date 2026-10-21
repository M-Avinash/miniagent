"""app.py — Northwind OData QnA chatbot entry point.

Run: python -m miniagent.examples.northwind.app
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from miniagent import HandoffOrchestrator

from .schema import fetch_schema, _schema
from .tools import shared_state
from .agents import (
    create_orchestrator_agent,
    create_data_fetcher_agent,
    create_data_analyst_agent,
    graph_memory,
)


def main():
    print("=" * 60)
    print("Northwind OData QnA Chatbot (multi-agent)")
    print("Ask questions about the Northwind database.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    print("Fetching schema from Northwind OData service...")
    fetch_schema()
    if _schema:
        for entity_set, fields in _schema.items():
            print(f"  {entity_set}: {len(fields)} fields")
    else:
        print("  (schema fetch failed — using fallback)")
    print()

    orchestrator = create_orchestrator_agent()
    data_fetcher = create_data_fetcher_agent()
    data_analyst = create_data_analyst_agent()

    orch = HandoffOrchestrator(
        [orchestrator, data_fetcher, data_analyst],
        max_handoffs=5,
        agent_names={
            "orchestrator": orchestrator,
            "data_fetcher": data_fetcher,
            "data_analyst": data_analyst,
        },
        shared_state=shared_state,
    )

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "reset":
            shared_state["entities"].clear()
            shared_state["data_desc"] = ""
            orch.reset_session()
            graph_memory.clear()
            data_analyst.reset_analyze_attempts()
            print("[context reset]")
            continue

        data_analyst.reset_analyze_attempts()

        if graph_memory.path:
            print(f"  [graph] {graph_memory.focus_path()}")

        print()
        try:
            result = orch.run(question)
        except Exception as e:
            print(f"[error: {type(e).__name__}: {e}]")
            continue

        print()
        print("-" * 60)
        print(f"Answer: {result.text}")
        if result.aborted:
            print(f"(aborted: {result.abort_reason})")
        print("-" * 60)


if __name__ == "__main__":
    main()
