"""Northwind OData QnA chatbot — interactive multi-agent terminal app.

Architecture (handoff orchestrator):
  1. Orchestrator Agent — understands the question, decides which agent to hand off to
  2. Data Fetcher Agent — builds and executes OData queries, stores results in shared state
  3. Data Analyst Agent — runs Python against the shared data, computes the answer

Run it:
    python -m miniagent.northwind_qna

Ask questions in the terminal. Type 'quit' to exit.
"""

import json
import sys

import requests

from . import (
    Agent,
    tool,
    HandoffOrchestrator,
    EventType,
    verbose_log,
    run_python,
)

NORTHWIND_URL = "https://services.odata.org/V4/Northwind/Northwind.svc"

# Shared state: agents pass data to each other through this store, not through
# tool-call arguments (which would overflow a local model's JSON buffer).
shared_state = {"data": None, "data_desc": ""}


# --------------------------------------------------------------------- tools
@tool
def odata_query(entity: str, filter: str = None, select: str = None,
                top: int = None, skip: int = None, orderby: str = None,
                expand: str = None) -> str:
    """Query the Northwind OData service.
    Args:
        entity (str): The entity set to query (e.g., 'Orders', 'Products', 'Customers').
        filter (str): OData $filter expression (e.g., "ShipCountry eq 'Germany'").
        select (str): OData $select expression (e.g., "OrderID,OrderDate,ShippedDate").
        top (int): OData $top (limit the number of results).
        skip (int): OData $skip (offset for pagination).
        orderby (str): OData $orderby expression (e.g., "OrderDate desc").
        expand (str): OData $expand expression (e.g., "Order_Details($expand=Products)").
    """
    params = {}
    if filter:
        params["$filter"] = filter
    if select:
        params["$select"] = select
    if top is not None:
        params["$top"] = top
    if skip is not None:
        params["$skip"] = skip
    if orderby:
        params["$orderby"] = orderby
    if expand:
        params["$expand"] = expand

    url = f"{NORTHWIND_URL}/{entity}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Store the full result in shared state so the analyst can use it
        # without it passing through a tool-call argument.
        shared_state["data"] = data
        count = len(data.get("value", []))
        shared_state["data_desc"] = f"{count} records from {entity}"
        # Return a short summary + a sample, not the full JSON (keeps context small).
        sample = data.get("value", [])[:3]
        return json.dumps({
            "status": "ok",
            "count": count,
            "sample": sample,
            "note": f"Full result ({count} records) stored in shared state for the analyst.",
        })
    except requests.RequestException as e:
        return f"[error: {e}]"
    except json.JSONDecodeError as e:
        return f"[error: invalid JSON: {e}]"


@tool
def odata_metadata() -> str:
    """Fetch the Northwind OData service metadata (entity list and schema)."""
    try:
        resp = requests.get(NORTHWIND_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Return a compact list of entity sets, not the full verbose metadata.
        entities = [e.get("name") for e in data.get("EntitySets", [])]
        return json.dumps({"entity_sets": entities})
    except requests.RequestException as e:
        return f"[error: {e}]"


@tool
def analyze_data(code: str) -> str:
    """Run Python code against the data stored in shared state.
    The variable `data` is pre-loaded with the full result from odata_query.
    Print your answer with print() — it will be captured and returned.
    Args:
        code (str): Python code to run. Use `data` to access the fetched records.
    """
    if shared_state["data"] is None:
        return "[error: no data in shared state — call odata_query first]"
    data = shared_state["data"]
    # Run the code in a namespace that has `data` available.
    namespace = {"data": data}
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, namespace)
    except Exception as e:
        return f"[error: {type(e).__name__}: {e}]"
    out = buf.getvalue().strip()
    if not out:
        return "[no output — use print() to show your answer]"
    return out


# ------------------------------------------------------------------- agents
def create_orchestrator_agent() -> Agent:
    """The orchestrator agent — understands the question and delegates."""
    system_prompt = """You are an orchestrator for a QnA chatbot on the Northwind OData service.

Your job:
1. Understand the user's question.
2. Decide which agent to hand off to:
   - If the question needs data from the OData service, hand off to "data_fetcher".
   - If you can answer directly (a clarification, a greeting, or a question about the data schema), answer it yourself.
3. When you hand off, include the specific context the next agent needs (which entity, what filter, what to compute).

IMPORTANT: To hand off, you MUST include the exact string "HANDOFF: data_fetcher" in your final answer.
Example: "I need orders shipped to Germany in Q2 1997 with their ship dates. HANDOFF: data_fetcher"

The data spans 1996-1997. "Last quarter" means Q2 1997 (April-June 1997)."""

    def handoff_hook(event):
        if event.type == EventType.AGENT_END:
            text = event.data.get("final_text", "")
            if "HANDOFF: data_fetcher" in text:
                event.data["handoff"] = "data_fetcher"
        return event

    return Agent(
        system_prompt,
        hooks=[handoff_hook, verbose_log("orchestrator")],
        name="orchestrator",
        max_tool_calls=5,
    )


def create_data_fetcher_agent() -> Agent:
    """The data fetcher agent — builds and executes OData queries."""
    system_prompt = """You are a data fetcher for the Northwind OData service.

Your job:
1. Take the request (which entity, what filter, what fields).
2. Build the correct OData query and call odata_query.
3. odata_query stores the full result in shared state and returns a summary.

OData $filter syntax:
- String compare: ShipCountry eq 'Germany'
- Date range: OrderDate ge 1997-04-01T00:00:00Z and OrderDate lt 1997-07-01T00:00:00Z
- Combine: ShipCountry eq 'Germany' and OrderDate ge 1997-04-01T00:00:00Z

Date format: YYYY-MM-DDTHH:MM:SSZ

Common entities:
- Orders: OrderID, CustomerID, OrderDate, RequiredDate, ShippedDate, ShipCountry, Freight
- Products: ProductID, ProductName, CategoryID, UnitPrice, UnitsInStock
- Customers: CustomerID, CompanyName, Country, City, ContactName

After you fetch the data, always hand off to the analyst by including the exact
string "HANDOFF: data_analyst" in your final answer, plus a one-line note about
what you fetched and what the analyst should compute."""

    def handoff_hook(event):
        if event.type == EventType.AGENT_END:
            event.data["handoff"] = "data_analyst"
        return event

    return Agent(
        system_prompt,
        tools=[odata_query, odata_metadata],
        hooks=[handoff_hook, verbose_log("data_fetcher")],
        name="data_fetcher",
        max_tool_calls=10,
    )


def create_data_analyst_agent() -> Agent:
    """The data analyst agent — runs Python against shared data, computes the answer."""
    system_prompt = """You are a data analyst for the Northwind OData service.

The data has been fetched and stored in shared state. You have a tool `analyze_data`
that runs Python code where the variable `data` is the full OData result
(data['value'] is the list of records).

Your job:
1. Write Python code that computes the answer the user asked for.
2. Call analyze_data with that code. Use print() to show the answer.
3. Return a clear, concise final answer in natural language.

Useful patterns:
- Count: len(data['value'])
- Date diff: (datetime.fromisoformat(r['ShippedDate'].replace('Z','+00:00')) - datetime.fromisoformat(r['OrderDate'].replace('Z','+00:00'))).days
- Group by: from collections import Counter; Counter(r['ShipCountry'] for r in data['value'])
- Filter: [r for r in data['value'] if r['ShipCountry'] == 'Germany']

Do NOT set a handoff. Your answer is the final answer."""

    return Agent(
        system_prompt,
        tools=[analyze_data],
        hooks=[verbose_log("data_analyst")],
        name="data_analyst",
        max_tool_calls=10,
    )


# ------------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("Northwind OData QnA Chatbot (multi-agent)")
    print("Ask questions about the Northwind database.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    # Create the agents
    orchestrator = create_orchestrator_agent()
    data_fetcher = create_data_fetcher_agent()
    data_analyst = create_data_analyst_agent()

    # Create the orchestrator
    orch = HandoffOrchestrator(
        [orchestrator, data_fetcher, data_analyst],
        max_handoffs=5,
        agent_names={
            "orchestrator": orchestrator,
            "data_fetcher": data_fetcher,
            "data_analyst": data_analyst,
        },
    )

    # Interactive loop
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

        # Clear shared state for each new question
        shared_state["data"] = None
        shared_state["data_desc"] = ""

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
