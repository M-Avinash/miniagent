"""Northwind OData QnA chatbot — interactive multi-agent terminal app.

Architecture (handoff orchestrator):
  1. Orchestrator Agent — understands the question, decides which agent to hand off to
  2. Data Fetcher Agent — builds and executes OData queries, stores results in shared state
  3. Data Analyst Agent — runs Python against the shared data, computes the answer

Ask questions in the terminal. Type 'quit' to exit.

Module-level state:
  - shared_state: agents pass data to each other through this store rather than
    through tool-call arguments (which would overflow a local model's JSON buffer).
  - graph_memory: tracks entity keys and navigation paths across multi-turn
    conversations, preventing context drift and key hallucination.
  - ENTITY_KEY_MAP: entity -> key field mapping for the graph memory hook.
  - _schema: dynamic schema (entity set -> field names) fetched from the OData
    service at startup so the prompts always reflect the real field names.
"""

import json
import os
import re
import sys
import tempfile

# Allow running from inside the miniagent folder: python examples/northwind_qna.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import requests

from miniagent import (
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

shared_state: dict = {"entities": {}, "data_desc": ""}
graph_memory: GraphMemory = GraphMemory()

ENTITY_KEY_MAP: dict = {
    "Order": "OrderID",
    "Customer": "CustomerID",
    "Product": "ProductID",
    "Employee": "EmployeeID",
}

_schema: dict = {}


def _entity_type_for(entity_set: str) -> str:
    """Map an entity set name (plural) to its entity type name (singular).
    e.g. 'Orders' -> 'Order', 'Products' -> 'Product'.
    Falls back to the set name itself if no singular form is known.
    """
    if entity_set.endswith("s") and not entity_set.endswith("ss"):
        return entity_set[:-1]
    return entity_set


def fetch_schema(entities: list = None) -> dict:
    """Fetch the real field list for each entity from the OData $metadata.

    Returns a dict mapping entity set name -> list of field names.
    Caches the result in _schema. On failure, returns the current _schema
    (possibly empty) so the caller can fall back to a hardcoded schema.
    """
    if entities is None:
        entities = ["Orders", "Products", "Customers", "Employees", "Categories", "Order_Details"]
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


FIELD_DESCRIPTIONS = {
    "Orders": {
        "OrderID": "Unique order identifier",
        "CustomerID": "ID of the customer who placed the order (links to Customers.CustomerID)",
        "EmployeeID": "ID of the employee who took/handled the order (links to Employees.EmployeeID)",
        "OrderDate": "Date the order was placed",
        "RequiredDate": "Date the order is required to be shipped by (delivery deadline)",
        "ShippedDate": "Date the order was actually shipped (may be null if not yet shipped)",
        "ShipVia": "ID of the shipper/carrier used for delivery",
        "Freight": "Shipping cost charged to the customer (in USD)",
        "ShipName": "Name of the shipping recipient / shipping partner / shipper",
        "ShipAddress": "Street address of the shipping destination",
        "ShipCity": "City of the shipping destination",
        "ShipRegion": "Region/state/province of the shipping destination",
        "ShipPostalCode": "Postal/ZIP code of the shipping destination",
        "ShipCountry": "Country of the shipping destination",
    },
    "Products": {
        "ProductID": "Unique product identifier",
        "ProductName": "Name of the product",
        "SupplierID": "ID of the supplier who provides this product",
        "CategoryID": "ID of the product category (links to Categories.CategoryID)",
        "QuantityPerUnit": "Package size/description (e.g., '10 - 500 g cans')",
        "UnitPrice": "Price per unit (in USD)",
        "UnitsInStock": "Number of units currently in stock (0 = out of stock)",
        "UnitsOnOrder": "Number of units ordered from supplier but not yet received",
        "ReorderLevel": "Stock level at which reorder alerts are triggered",
        "Discontinued": "True if the product is no longer available",
    },
    "Customers": {
        "CustomerID": "Unique customer identifier (links to Orders.CustomerID)",
        "CompanyName": "Name of the customer company",
        "ContactName": "Name of the primary contact person",
        "ContactTitle": "Job title of the contact person",
        "Address": "Street address of the customer",
        "City": "City of the customer",
        "Region": "Region/state/province of the customer",
        "PostalCode": "Postal/ZIP code of the customer",
        "Country": "Country of the customer",
        "Phone": "Phone number of the customer",
        "Fax": "Fax number of the customer",
    },
    "Employees": {
        "EmployeeID": "Unique employee identifier (links to Orders.EmployeeID)",
        "LastName": "Employee last name",
        "FirstName": "Employee first name",
        "Title": "Job title",
        "TitleOfCourtesy": "Courtesy title (Mr., Mrs., Dr., etc.)",
        "BirthDate": "Date of birth",
        "HireDate": "Date the employee was hired",
        "Address": "Home address",
        "City": "Home city",
        "Region": "Home region/state",
        "PostalCode": "Home postal code",
        "Country": "Home country",
        "HomePhone": "Home phone number",
        "Extension": "Office phone extension",
        "Photo": "Employee photo (binary, ignore)",
        "Notes": "Internal notes about the employee",
        "ReportsTo": "EmployeeID of the manager this employee reports to",
        "PhotoPath": "File path to employee photo",
    },
    "Categories": {
        "CategoryID": "Unique category identifier (links to Products.CategoryID)",
        "CategoryName": "Name of the product category",
        "Description": "Description of the category",
        "Picture": "Category image (binary, ignore)",
    },
    "Order_Details": {
        "OrderID": "Order this line item belongs to (links to Orders.OrderID)",
        "ProductID": "Product in this line item (links to Products.ProductID)",
        "UnitPrice": "Price per unit at time of order",
        "Quantity": "Number of units ordered",
        "Discount": "Discount applied to this line item (0 to 1)",
    },
}


def schema_prompt_block() -> str:
    """Build a prompt-ready string of the schema with field descriptions."""
    if not _schema:
        return "(schema unavailable — call odata_metadata to discover fields)"
    lines = []
    for entity_set, fields in _schema.items():
        descs = FIELD_DESCRIPTIONS.get(entity_set, {})
        field_parts = []
        for f in fields:
            desc = descs.get(f, "")
            if desc:
                field_parts.append(f"{f} ({desc})")
            else:
                field_parts.append(f)
        lines.append(f"- {entity_set}: {', '.join(field_parts)}")
    return "\n".join(lines)


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
        shared_state["entities"][entity] = data
        count = len(data.get("value", []))
        shared_state["data_desc"] = ", ".join(
            f"{k}: {len(v.get('value', []))} records" for k, v in shared_state["entities"].items()
        )
        sample = data.get("value", [])[:3]
        return json.dumps({
            "status": "ok",
            "count": count,
            "sample": sample,
            "note": f"Full result ({count} records from {entity}) stored in shared state.",
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
    entities = shared_state["entities"]
    if not entities:
        return "[error: no data in shared state — call odata_query first]"

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entities, f)
        preamble = (
            "import json as _json\n"
            f"with open({path!r}) as _f:\n"
            "    _entities = _json.load(_f)\n"
        )
        # Create a variable for each entity: Orders, Products, Customers, etc.
        for name in entities:
            preamble += f"    {name} = _entities['{name}']['value']\n"
        # Also create a generic 'data' variable pointing to the last entity
        last_entity = list(entities.keys())[-1]
        preamble += f"    data = {last_entity}\n"
        full_code = preamble + code
        result = run_python({"code": full_code, "timeout": timeout})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if result.startswith("[exit 0]"):
        body = result[len("[exit 0]"):].strip()
        return body if body else "[no output — use print() to show your answer]"
    return result


def create_orchestrator_agent() -> Agent:
    """The orchestrator agent — understands the question and delegates."""
    system_prompt = """You are an orchestrator for a QnA chatbot on the Northwind OData service.

Your job:
1. Understand the user's question. Check your conversation history for previous
   Q&A rounds — the user may reference "that group", "those orders", "the top
   customer", etc. Resolve these references using the previous answers.
2. Decide which agent to hand off to:
   - If the question needs data from the OData service, hand off to "data_fetcher".
   - If you can answer directly (a clarification, a greeting, or a question
     about the data schema), answer it yourself.
3. When you hand off, include the specific context the next agent needs:
   - Which entity to fetch (Orders, Products, Customers, etc.)
   - What to compute (count, sum, average, list)
   - Any references from previous answers (e.g., "the UK customers are AROUT,
     BSBEV, CONSH, EASTC — filter orders to those CustomerIDs")

To hand off, call the handoff tool with target_agent='data_fetcher'.
Example: call handoff(target_agent='data_fetcher') with context like
"I need orders shipped to Germany in Q1 1997 with their ship dates."

The data spans July 1996 to February 1997 (200 orders). "Last quarter" means
Q1 1997 (January-February 1997)."""

    return Agent(
        system_prompt,
        tools=[make_handoff_tool(["data_fetcher"])],
        hooks=[handoff_hook, verbose_log("orchestrator")],
        name="orchestrator",
        max_tool_calls=5,
    )


def create_data_fetcher_agent() -> Agent:
    """The data fetcher agent — builds and executes OData queries."""
    schema_block = schema_prompt_block()

    system_prompt = f"""You are a data fetcher for the Northwind OData service.

Your job: translate the user's question into an OData query, fetch the data,
and hand off to the analyst.

## Rules

1. ALWAYS fetch the entity with NO $filter parameter. Call odata_query with
   just the entity name and no filter. The service returns different datasets
   depending on the filter used, which makes results inconsistent. Even if the
   handoff context says "orders to Germany" or "in 1997", do NOT add those as
   OData filters. Fetch the full entity and let the analyst do ALL filtering
   (date, country, category, numeric thresholds) in Python.
   The full Orders dataset spans July 1996 to February 1997 (200 orders).

2. Do NOT put ANY conditions in the OData $filter — not country, not category,
   not date, not numeric thresholds. The service returns different datasets
   depending on the filter, which makes results inconsistent. Always fetch the
   full entity with no filter and let the analyst do ALL filtering in Python.

4. Fetch the FULL entity (do not use $select) so the analyst has every field.

5. If the question references data from multiple entities, you MUST fetch ALL of
   them in separate odata_query calls. Each entity is stored in shared state
   under its own name (e.g., `Orders`, `Products`, `Categories`). The analyst
   can then access them as variables in Python.
   Examples:
   - "products in the Beverages category" → fetch Products AND Categories
   - "which country has the most customers" → fetch Customers AND Orders
   - "total revenue by product" → fetch Order_Details AND Products AND Orders
   Note in your handoff context exactly which entities you fetched.

6. If the question is about a single entity (e.g. "all products"), fetch just
   that entity. Do NOT fetch unrelated entities.

## Schema (entity set -> fields with descriptions)
{schema_block}

## Domain rules
- "Delayed" means ShippedDate gt RequiredDate (or ShippedDate is null).
- "Q1/Q2/Q3/Q4" = Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec.
- The data spans July 1996 to February 1997 (200 orders total: 152 in 1996,
  48 in 1997). "Last quarter" means Q1 1997.

## Handoff

After fetching, call handoff(target_agent='data_analyst') with a one-line note:
what you fetched (entity + filter) and what the analyst should compute.
Example: "Fetched Orders where ShipCountry eq 'Germany' and OrderDate in 1997.
Compute: count, avg freight, and which customer had highest total freight."""

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
    schema_block = schema_prompt_block()

    system_prompt = f"""You are a data analyst for the Northwind OData service.

The data has been fetched and stored in shared state. You have a tool `analyze_data`
that runs Python code where each fetched entity is available as a variable:
  - Orders: list of order records
  - Products: list of product records
  - Customers: list of customer records
  - Categories: list of category records
  - Order_Details: list of order line-item records
  - data: alias for the last entity fetched (backward compat)
Each variable is a list of dicts — filter, group, or compute on any field.

SESSION CONTEXT:
Your user message may include "Previous Q&A in this session:" with earlier
questions and answers. Use this to resolve references like "that group",
"those orders", "the top customer", etc.
Example: if a previous answer was "UK customers: AROUT, BSBEV, CONSH, EASTC"
and the question is "how many orders did that group place in 1997?", filter
Orders to CustomerID in ['AROUT','BSBEV','CONSH','EASTC'] and year==1997.
The entities available in shared state are listed in the handoff context.

Schema (entity set -> fields):
{schema_block}

Domain rules:
- "Delayed" means ShippedDate > RequiredDate (or ShippedDate is null).
- "Q1/Q2/Q3/Q4" = Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec.
- The data spans July 1996 to February 1997 (200 orders). "Last quarter" means Q1 1997 (January-February 1997).

Your job:
1. Write Python code that computes the answer the user asked for.
2. Call analyze_data with that code. Use print() to show the answer.
3. If the code fails (you see an [error: ...] or [timeout] result), READ the
   error, fix the code, and call analyze_data again. Retry up to 3 times.
4. Return a clear, concise final answer in natural language.

CRITICAL RULES — NO HALLUCINATIONS:
- Your final answer MUST match exactly what your Python code printed.
  If the code printed 0, say 0. If it printed an empty list, say none.
- Do NOT estimate, guess, or round numbers that the code did not produce.
- If the data does not contain what the question asks for, say so explicitly.
  Do not fabricate an answer to seem helpful.
- If you are uncertain whether your code computed the right thing, re-run
  it with a simpler check before giving your final answer.

DATE FILTERING:
- The data in shared state may contain orders from multiple years.
  ALWAYS filter by the relevant date range in Python before computing.
  Example: if the question says "in 1997", filter to:
    [r for r in data['value'] if datetime.fromisoformat(r['OrderDate'].replace('Z','+00:00')).year == 1997]
  If it says "Q1 1997", filter to year==1997 AND month in (1,2,3).
  Never compute over the full dataset when the question specifies a time period.

Useful patterns:
- Count: len(data['value'])
- Date parse: datetime.fromisoformat(r['ShippedDate'].replace('Z','+00:00'))
- Date diff: (ship_dt - order_dt).days
- Group by: from collections import Counter; Counter(r['ShipCountry'] for r in data['value'])
- Filter: [r for r in data['value'] if r['ShipCountry'] == 'Germany']
- Delayed: [r for r in data['value'] if r.get('ShippedDate') and r.get('RequiredDate') and parse(r['ShippedDate']) > parse(r['RequiredDate'])]

Do NOT set a handoff. Your answer is the final answer."""

    attempts = {"count": 0}

    def self_heal_hook(event):
        """Bound the self-healing retry loop.

        When analyze_data returns an error, count it. After 3 failed attempts,
        set an abort so the agent stops retrying and returns whatever it has.
        The counter is reset at the start of each new question (see main()).
        """
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

    agent.reset_analyze_attempts = lambda: attempts.update(count=0)
    return agent


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
