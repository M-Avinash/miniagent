"""agents.py — the three agent factories for the Northwind QnA example."""

from miniagent import (
    Agent,
    EventType,
    verbose_log,
    make_handoff_tool,
    handoff_hook,
    GraphMemory,
    graph_memory_hook,
    graph_injection_hook,
)

from .schema import schema_prompt_block
from .tools import odata_query, odata_metadata, analyze_data, shared_state

graph_memory: GraphMemory = GraphMemory()

ENTITY_KEY_MAP: dict = {
    "Order": "OrderID",
    "Customer": "CustomerID",
    "Product": "ProductID",
    "Employee": "EmployeeID",
}


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
