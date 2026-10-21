# Northwind Multi-Agent Chatbot

A terminal chatbot that answers natural-language questions against the SAP Northwind OData service. You type a question, it fetches live data, runs Python analysis, and returns an answer.

```
Orchestrator ──handoff──▶ Data Fetcher ──handoff──▶ Data Analyst
```

**Three agents, each with a single job:**

| Agent | Tools | Framework pieces used |
|-------|-------|----------------------|
| Orchestrator | `handoff` | `make_handoff_tool`, `handoff_hook` — reads the question, routes to the fetcher |
| Data Fetcher | `odata_query`, `odata_metadata`, `handoff` | `@tool` (OData queries), `GraphMemory` hooks (tracks entity keys), shared state (stores full result) |
| Data Analyst | `analyze_data` | `run_python` sandbox (executes LLM-generated Python against the data), self-healing hook (retries up to 3x on error) |

**How the framework pieces fit together in this example:**

- **Agentic loop** (`agent.py`) — each agent runs its own loop, calling tools until it produces a final answer or hands off.
- **Handoff orchestrator** (`orchestrator.py`) — follows the chain. The orchestrator agent calls `handoff(target_agent='data_fetcher')`, the hook sets the route, the orchestrator runs the next agent.
- **Shared state** — a dict passed to `HandoffOrchestrator`. The fetcher stores fetched entities (Orders, Products, etc.); the analyst reads them via `analyze_data`. The orchestrator maintains a `session` list for multi-turn context. Avoids passing large JSON through tool-call arguments.
- **Graph memory** (`graph_memory.py`) — after each `odata_query`, a hook extracts primary keys (e.g. `OrderID: 10248`) and stores them. Before each LLM call, another hook injects the active entity path into the system prompt so the model knows which records are in scope.
- **Sandbox** (`sandbox.py`) — the analyst writes Python code, `run_python` executes it in an isolated subprocess (minimal env, temp cwd, timeout). Errors come back as strings so the agent can fix and retry.
- **Self-healing hook** — counts `analyze_data` failures. After 3 errors, sets `abort` so the agent stops retrying and returns what it has.
- **Verbose log** (`observe.py`) — every LLM call, tool call, and result is printed to the terminal, labeled by agent name.

**Multi-turn:** `shared_state` persists across questions in a session. The `HandoffOrchestrator` maintains a `session` list in shared state that accumulates Q&A history. Before each agent runs, the orchestrator injects the session context into the user message, so every agent sees previous questions and answers. This is deterministic — no agent needs to know it's multi-turn. You can ask follow-ups ("of those, how many had freight over 100?") without re-fetching. Type `reset` to clear context, `quit` to exit.

**Multi-entity:** The fetcher can store multiple entities in shared state (e.g., `Orders` and `Products`). The analyst accesses them as Python variables in `analyze_data`.

```
You: How many orders were shipped to Germany in Q1 1997, and how many were
     delivered within 3 weeks of the order date?

[01:27:16] [orchestrator] START: How many orders were shipped to Germany in Q1 1997...
[01:27:16] [orchestrator] TOOL: handoff({'target_agent': 'data_fetcher'})
[01:27:17] [data_fetcher] TOOL: odata_query({'entity': 'Orders'})
[01:27:18] [data_fetcher] RESULT: {"status": "ok", "count": 200, ...}
[01:27:19] [data_fetcher] TOOL: handoff({'target_agent': 'data_analyst'})
[01:27:21] [data_analyst] TOOL: analyze_data({'code': "from datetime import datetime..."})
[01:27:21] [data_analyst] RESULT: {'count': 4, 'within_3_weeks': 3, 'longer': [10407]}

Answer: In Q1 1997, 4 orders were shipped to Germany. 3 were delivered
        within 3 weeks. Order 10407 took 23 days.
```

## Data Source

The example queries the SAP Northwind OData V4 demo service:

```
https://services.odata.org/V4/Northwind/Northwind.svc
```

- **Version:** OData V4
- **Entities:** Orders (200), Products (20), Customers (20), Employees (9), Categories (8), Order_Details (196)
- **Date range:** Orders span July 1996 to February 1997

To cross-check any answer, query the service directly:

```
# All orders
https://services.odata.org/V4/Northwind/Northwind.svc/Orders

# Orders with a filter
https://services.odata.org/V4/Northwind/Northwind.svc/Orders?$filter=ShipCountry%20eq%20%27Germany%27
```

## Run

```bash
python -m miniagent.examples.northwind.app
```

Test questions: [`Northwind_test_questions.md`](../Northwind_test_questions.md)
