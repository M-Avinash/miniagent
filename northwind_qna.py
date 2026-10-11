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
import os
import re
import sys
import tempfile

import requests

from . import (
    Agent,
    tool,
    HandoffOrchestrator,
    EventType,
    verbose_log,
    run_python,
    make_handoff_tool,
    handoff_hook,
    GraphMemory,
    graph_memory_hook,
    graph_injection_hook,
)

NORTHWIND_URL = "https://services.odata.org/V4/Northwind/Northwind.svc"

# Shared state: agents pass data to each other through this store, not through
# tool-call arguments (which would overflow a local model's JSON buffer).
shared_state = {"data": None, "data_desc": ""}

# Semantic graph memory: tracks entity keys and navigation paths across
# multi-turn conversations. Prevents context drift and key hallucination.
graph_memory = GraphMemory()

# Entity -> key field mapping for the graph memory hook.
ENTITY_KEY_MAP = {
    "Order": "OrderID",
    "Customer": "CustomerID",
    "Product": "ProductID",
    "Employee": "EmployeeID",
}

# Dynamic schema: fetched from the OData service at startup so the prompts
# always reflect the real field names (not a hardcoded copy).
# Maps entity set name -> list of field names.
_schema: dict = {}


def _entity_type_for(entity_set: str) -> str:
    """Map an entity set name (plural) to its entity type name (singular).
    e.g. 'Orders' -> 'Order', 'Products' -> 'Product'.
    Falls back to the set name itself if no singular form is known.
    """
    # Simple heuristic: strip trailing 's' for common cases.
    if entity_set.endswith("s") and not entity_set.endswith("ss"):
        return entity_set[:-1]
    return entity_set


def fetch_schema(entities: list = None) -> dict:
    """Fetch the real field list for each entity from the OData $metadata.

    Returns a dict mapping entity set name -> list of field names.
    Caches the result in _schema. On failure, returns the current _schema
    (possibly empty) so the caller can fall back to a hardcoded schema.
    """
    global _schema
    if entities is None:
        entities = ["Orders", "Products", "Customers", "Employees", "Categories"]
    try:
        resp = requests.get(f"{NORTHWIND_URL}/$metadata", timeout=30)
        resp.raise_for_status()
        text = resp.text
        for entity_set in entities:
            entity_type = _entity_type_for(entity_set)
            pattern = r'<EntityType Name="' + entity_type + r'">.*?</EntityType>'
            m = re.search(pattern, text, re.DOTALL)
            if m:
                fields = re.findall(r'<Property Name="([^"]+)"', m.group(0))
                if fields:
                    _schema[entity_set] = fields
    except requests.RequestException:
        pass  # fall back to whatever _schema already has (or empty)
    return _schema


def schema_prompt_block() -> str:
    """Build a prompt-ready string of the schema, e.g.:
        Orders: OrderID, CustomerID, EmployeeID, ...
    """
    if not _schema:
        return "(schema unavailable — call odata_metadata to discover fields)"
    lines = []
    for entity_set, fields in _schema.items():
        lines.append(f"- {entity_set}: {', '.join(fields)}")
    return "\n".join(lines)


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
def analyze_data(code: str, timeout: int = 30) -> str:
    """Run Python code against the data stored in shared state, in a sandbox.

    The code runs in a subprocess with a timeout (reuses run_python). The
    variable `data` is pre-loaded with the full result from odata_query.
    Print your answer with print() — it will be captured and returned.

    If the code fails, the error (with traceback) is returned so the agent
    can fix the code and retry (self-healing).

    Args:
        code (str): Python code to run. Use `data` to access the fetched records.
        timeout (int): Max seconds before the subprocess is killed.
    """
    if shared_state["data"] is None:
        return "[error: no data in shared state — call odata_query first]"

    data = shared_state["data"]
    # Write the full data to a temp file so the subprocess can load it.
    # This keeps the data out of the code string (and out of the LLM's context).
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        # Prepend a loader so the generated code can use `data` directly.
        preamble = (
            "import json as _json\n"
            f"with open({path!r}) as _f:\n"
            "    data = _json.load(_f)\n"
        )
        full_code = preamble + code
        # Reuse the sandboxed runner: subprocess + timeout, errors as strings.
        result = run_python({"code": full_code, "timeout": timeout})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    # Strip the [exit 0] marker for a cleaner return; keep error/timeout markers.
    if result.startswith("[exit 0]"):
        body = result[len("[exit 0]"):].strip()
        return body if body else "[no output — use print() to show your answer]"
    return result


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

To hand off, call the handoff tool with target_agent='data_fetcher'.
Example: call handoff(target_agent='data_fetcher') with context like "I need orders shipped to Germany in Q2 1997 with their ship dates."

The data spans 1996-1997. "Last quarter" means Q2 1997 (April-June 1997)."""

    return Agent(
        system_prompt,
        tools=[make_handoff_tool(["data_fetcher"])],
        hooks=[handoff_hook, verbose_log("orchestrator")],
        name="orchestrator",
        max_tool_calls=5,
    )


def create_data_fetcher_agent() -> Agent:
    """The data fetcher agent — builds and executes OData queries."""
    # Build the schema block dynamically (real fields from the OData service).
    schema_block = schema_prompt_block()

    system_prompt = f"""You are a data fetcher for the Northwind OData service.

Your job:
1. Take the request (which entity, what filter, what to compute).
2. Build the correct OData query and call odata_query.
3. odata_query stores the FULL result in shared state and returns a summary.
   Fetch the FULL entity (do not use $select) so the analyst has every field.

OData $filter syntax:
- String compare: ShipCountry eq 'Germany'
- Number compare: Freight gt 100
- Date range: OrderDate ge 1997-04-01T00:00:00Z and OrderDate lt 1997-07-01T00:00:00Z
- Combine with and/or: ShipCountry eq 'Germany' and OrderDate ge 1997-04-01T00:00:00Z
- In list: ShipCountry in ('Germany','France','Italy')
- Not equal: ShipCountry ne 'Germany'

Date format: YYYY-MM-DDTHH:MM:SSZ

Complex boolean examples (use parentheses to group):
- (ShipCountry eq 'Germany' or ShipCountry eq 'France') and Freight gt 100
- ShipCountry eq 'Germany' and (ShipVia eq 1 or ShipVia eq 2)
- not (Discontinued eq true)

Schema (entity set -> fields):
{schema_block}

Domain rules:
- "Delayed" means ShippedDate gt RequiredDate (or ShippedDate is null).
- "Q1/Q2/Q3/Q4" = Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec.
- The data spans 1996-1997. "Last quarter" means Q2 1997 (April-June 1997).

After you fetch the data, always hand off to the analyst by calling the handoff
tool with target_agent='data_analyst', plus a one-line note about what you
fetched and what the analyst should compute."""

    return Agent(
        system_prompt,
        tools=[odata_query, odata_metadata, make_handoff_tool(["data_analyst"])],
        hooks=[
            graph_memory_hook(graph_memory, ENTITY_KEY_MAP),
            graph_injection_hook(graph_memory),
            handoff_hook,
            verbose_log("data_fetcher"),
        ],
        name="data_fetcher",
        max_tool_calls=10,
    )


def create_data_analyst_agent() -> Agent:
    """The data analyst agent — runs Python against shared data, computes the answer."""
    # Build the schema block dynamically (real fields from the OData service).
    schema_block = schema_prompt_block()

    system_prompt = f"""You are a data analyst for the Northwind OData service.

The data has been fetched and stored in shared state. You have a tool `analyze_data`
that runs Python code where the variable `data` is the full OData result
(data['value'] is the list of records). Each record is a FULL entity — every
field in the schema is present, so you can filter, group, or compute on any field.

Schema (entity set -> fields):
{schema_block}

Domain rules:
- "Delayed" means ShippedDate > RequiredDate (or ShippedDate is null).
- "Q1/Q2/Q3/Q4" = Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec.
- The data spans 1996-1997. "Last quarter" means Q2 1997 (April-June 1997).

Your job:
1. Write Python code that computes the answer the user asked for.
2. Call analyze_data with that code. Use print() to show the answer.
3. If the code fails (you see an [error: ...] or [timeout] result), READ the
   error, fix the code, and call analyze_data again. Retry up to 3 times.
4. Return a clear, concise final answer in natural language.

Useful patterns:
- Count: len(data['value'])
- Date parse: datetime.fromisoformat(r['ShippedDate'].replace('Z','+00:00'))
- Date diff: (ship_dt - order_dt).days
- Group by: from collections import Counter; Counter(r['ShipCountry'] for r in data['value'])
- Filter: [r for r in data['value'] if r['ShipCountry'] == 'Germany']
- Delayed: [r for r in data['value'] if r.get('ShippedDate') and r.get('RequiredDate') and parse(r['ShippedDate']) > parse(r['RequiredDate'])]

Do NOT set a handoff. Your answer is the final answer."""

    # Self-healing: when analyze_data returns an error, count it. After 3
    # failed attempts, set an abort so the agent stops retrying and returns
    # whatever it has. This bounds the retry loop. `attempts` is reset at the
    # start of each new question (see main()).
    attempts = {"count": 0}

    def self_heal_hook(event):
        if event.type == EventType.TOOL_RESULT and event.tool_name == "analyze_data":
            result = str(event.tool_result or "")
            if result.startswith("[error") or result.startswith("[timeout]"):
                attempts["count"] += 1
                if attempts["count"] >= 3:
                    event.data["abort"] = True
                    event.data["abort_reason"] = (
                        f"analyze_data failed {attempts['count']} times; giving up"
                    )
        return event

    agent = Agent(
        system_prompt,
        tools=[analyze_data],
        hooks=[self_heal_hook, verbose_log("data_analyst")],
        name="data_analyst",
        max_tool_calls=10,
    )

    # Expose a reset so main() can clear the retry counter between questions.
    agent.reset_analyze_attempts = lambda: attempts.update(count=0)
    return agent


# ------------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("Northwind OData QnA Chatbot (multi-agent)")
    print("Ask questions about the Northwind database.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    # Fetch the real schema from the OData service so the prompts use
    # accurate, up-to-date field names (not a hardcoded copy).
    print("Fetching schema from Northwind OData service...")
    fetch_schema()
    if _schema:
        for entity_set, fields in _schema.items():
            print(f"  {entity_set}: {len(fields)} fields")
    else:
        print("  (schema fetch failed — using fallback)")
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

    # Interactive loop (multi-turn: shared_state and graph_memory persist
    # across questions so the agent can reference prior context).
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

        # 'reset' clears the conversation context (shared state + graph).
        if question.lower() == "reset":
            shared_state["data"] = None
            shared_state["data_desc"] = ""
            graph_memory.clear()
            data_analyst.reset_analyze_attempts()
            print("[context reset]")
            continue

        # Reset the analyst's self-healing retry counter for each new question.
        data_analyst.reset_analyze_attempts()

        # Show the active graph context (if any) so the user can see what
        # the agent is tracking.
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
