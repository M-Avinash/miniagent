# miniagent

A minimal agent framework. Built with **Pi coding agent** and a self-hosted **Qwen3.8 27B LLM (llama.cpp, RTX 3090)**. The agentic loop lives in `agent.py`; everything else is a tool, a hook, or a wrapper. Runs on any **OpenAI-compatible API**. Scales from single agent to multi-agent handoff chains.

## Modules

| Module | Purpose |
|--------|---------|
| `agent.py` | Agentic loop (reason → act → observe), bounded by `max_tool_calls` |
| `llm.py` | Provider-agnostic LLM client, retry on 429 |
| `tool.py` | `@tool` decorator — type hints + docstring → JSON schema |
| `memory.py` | Bounded conversation window (sliding, system prompt pinned) |
| `hooks.py` | Event hooks — inspect, mutate, or abort at any loop step |
| `events.py` | Event types and schema definitions |
| `structured.py` | JSON extraction + field validation |
| `sandbox.py` | `run_python` — subprocess execution with timeout |
| `observe.py` | `verbose_log` hook — labeled event spans per agent |
| `observation.py` | ObservationalMemory — distilled cross-run JSONL store |
| `orchestrator.py` | HandoffOrchestrator (shared_state, session context), make_handoff_tool, handoff_hook |
| `graph_memory.py` | GraphMemory — entity key tracking for relational schemas |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│         Agentic Framework with orchestration               |   
│                                                            |
│                                                            │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐     │
│   │  Agent 1 │───────▶│  Agent 2 │───────▶│  Agent 3 │     │
│   │          │handoff │          │handoff │          │     │
│   │ • loop   │        │ • loop   │        │ • loop   │     │
│   │ • tools  │        │ • tools  │        │ • tools  │     │
│   │ • hooks  │        │ • hooks  │        │ • hooks  │     │
│   │ • memory │        │ • memory │        │ • memory │     │
│   └──────────┘        └──────────┘        └──────────┘     │
│        │                     │                     │       │
│        └─────────────────────┴─────────────────────┘       │
│                              │                             │
│                    Shared state (dict)                     │
└────────────────────────────────────────────────────────────┘
         │                                        ▲
         ▼                                        │
┌─────────────────────────────────────────────────────────────┐
│                        LLM Client                           │
│                    OpenAI-compatible API                    │
└─────────────────────────────────────────────────────────────┘
```

## Setup & Run

**Step 1:** Clone and install

```bash
git clone https://github.com/M-Avinash/miniagent
cd miniagent
pip install -r requirements.txt
```

**Step 2:** Create a file called `config.env` in the `miniagent` folder and save these details:

```
OPENAI_API_KEY=<your OpenAI API key>
MINIAGENT_MODEL=gpt-5.6-luna
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_TOOL_CHOICE=auto
MINIAGENT_REASONING_EFFORT=none
```

**Step 3:** Run the example

```bash
python examples/northwind_qna.py
```

**Step 4:** You're now in the chatbot. Type a question and press Enter. You can ask single-turn or multi-turn questions — Type `reset` to clear context, `quit` to exit.

Example questions are in [`Northwind_test_questions.md`](examples/Northwind_test_questions.md).

## Build Your Own Agent

An agent is an LLM client, a set of tools, and optional hooks. Tools are plain Python functions; the `@tool` decorator turns type hints and the docstring into a JSON schema.

Run this from inside the cloned `miniagent` folder (the repo folder must be named `miniagent`).

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))  # add repo root to path

from miniagent.agent import Agent
from miniagent.llm import LLMClient
from miniagent.tool import tool
from miniagent.observe import verbose_log

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny in {city}"

agent = Agent(
    name="weather_bot",
    llm=LLMClient(),
    system_prompt="You answer weather questions using your tools.",
    tools=[get_weather],
    hooks=[verbose_log("weather_bot")],
    max_tool_calls=5,
)

print(agent.run("What's the weather in Bengaluru?"))
```

For multi-agent chains, give each agent a `make_handoff_tool(...)` and run them with `HandoffOrchestrator`. See `examples/northwind_qna.py` for a full three-agent example.

## Example: Northwind Multi-Agent Chatbot

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
- **Sandbox** (`sandbox.py`) — the analyst writes Python code, `run_python` executes it in a subprocess with a timeout. Errors come back as strings so the agent can fix and retry.
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

See [`Northwind_test_questions.md`](examples/Northwind_test_questions.md) for test questions.

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

## Design Choices and Rationale

- **Minimal loop, everything else is a hook.** The reason → act → observe loop in `agent.py` stays small. Logging, entity tracking, self-healing and routing are all hooks, so behavior can be changed without touching the core.
- **Layered memory.** Each layer solves a different problem: a bounded window within a run, observational memory (distilled summaries) across runs, and graph memory for the records currently in scope.
- **Code execution instead of reasoning over raw data.** The analyst writes Python that runs in a sandboxed subprocess with a timeout. This is more accurate than asking the model to read rows, and errors come back as strings so the agent can self-correct.
- **Shared state instead of passing data through tool arguments.** Fetched datasets live in a dict that agents read directly, which avoids pushing large JSON through the context window.
- **Multi-agent handoffs with single-job agents.** Narrow tools and prompts per agent make routing explicit and each step easier to test.
- **Provider-agnostic.** Any **OpenAI-compatible** endpoint works, so the framework isn't tied to one vendor or to a local or hosted model.

## Development and Validation Approach

**Coding agent used:** Pi coding agent with a self-hosted Qwen3.8 27B LLM (llama.cpp, RTX 3090)

I worked with the coding agent as a pair programmer. I designed the architecture first (module layout, agentic loop, handoff mechanism), and then the LLM implemented each piece, which I reviewed. I validated each module with unit tests as we went, which also tied the different parts of the framework together.

I built the memory system in three stages, adding to it whenever I hit a limit. It started as plain conversation history: the system prompt, the user’s questions and the agent’s replies, capped at a fixed number of messages. Next, after each run, I had it save a short summary of what it was asked and what it answered to a JSONL file, and on the next run it reads those entries back. The third stage was a semantic graph. Once data is fetched, a hook pulls out the record IDs and builds a graph of what’s currently in scope, and that graph goes into the prompt so the model always knows which records it’s working with. I also changed the shared state to hold several data sets at once, so the analyst agent can use orders and products in the same question. The orchestrator keeps a running list of past questions and answers too, which is why a follow-up like “of those, how many had freight over 100?” works without re-fetching anything.

I validated the framework with 100+ unit tests covering every module: memory trimming, tool schema generation, sandbox timeouts, hook aborts, handoff routing, and session context injection. For the end-to-end check, I computed ground truth answers for 9 single-shot questions and 4 multi-turn conversations using pure Python queries against the OData service, with no LLM involved. I ran the agent against all of them and compared its responses to the ground truth. I kept iterating until every answer matched exactly.

