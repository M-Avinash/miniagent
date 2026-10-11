# miniagent

A minimal agent framework. Built with pi coding agent and a self-hosted Qwen3.8 27B LLM (llama.cpp, RTX 3090). The agentic loop lives in `agent.py`; everything else is a tool, a hook, or a wrapper. Runs on any OpenAI-compatible API. Scales from single agent to multi-agent handoff chains.

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
| `orchestrator.py` | HandoffOrchestrator, make_handoff_tool, handoff_hook |
| `graph_memory.py` | GraphMemory — entity key tracking for relational schemas |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ System     │  │ Tools      │  │ Hook         │   │
│  │ Prompt     │  │ (list)     │  │ Pipeline     │   │
│  └────────────┘  └────────────┘  └──────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │              Agent Loop (run)                  │ │
│  │                                                │ │
│  │  for turn in 1..max_tool_calls+1:              │ │
│  │    messages → LLM.chat(tools) → response       │ │
│  │    if response.tool_calls:                     │ │
│  │      for each tool_call:                       │ │
│  │        hooks → execute tool → append result    │ │
│  │      continue                                  │ │
│  │    else:                                       │ │
│  │      final_text = response.content; break      │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │         Memory (bounded list)                  │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    LLM Client       │  OpenAI-compatible API
│  (swappable)        │  (OpenAI, Ollama, vLLM, llama.cpp, ...)
└─────────────────────┘
```

## Setup & Run

**Step 1:** Clone and install

```bash
git clone https://github.com/M-Avinash/miniagent
cd miniagent
pip install -r requirements.txt
```

**Step 2:** Create a file called `config.env` in the root folder and save these details:

```
OPENAI_API_KEY=<your OpenAI API key>
MINIAGENT_MODEL=gpt-5.6-luna
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_TOOL_CHOICE=auto
MINIAGENT_REASONING_EFFORT=none
```

**Step 3:** Run the example

```bash
python -m miniagent.examples.northwind_qna
```

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
- **Shared state** — a module-level dict. The fetcher stores the full OData result; the analyst reads it via `analyze_data`. Avoids passing large JSON through tool-call arguments.
- **Graph memory** (`graph_memory.py`) — after each `odata_query`, a hook extracts primary keys (e.g. `OrderID: 10248`) and stores them. Before each LLM call, another hook injects the active entity path into the system prompt so the model knows which records are in scope.
- **Sandbox** (`sandbox.py`) — the analyst writes Python code, `run_python` executes it in a subprocess with a timeout. Errors come back as strings so the agent can fix and retry.
- **Self-healing hook** — counts `analyze_data` failures. After 3 errors, sets `abort` so the agent stops retrying and returns what it has.
- **Verbose log** (`observe.py`) — every LLM call, tool call, and result is printed to the terminal, labeled by agent name.

**Multi-turn:** `shared_state` and `GraphMemory` persist across questions in a session. You can ask follow-ups ("of those, how many had freight over 100?") without re-fetching. Type `reset` to clear context, `quit` to exit.

```bash
python -m miniagent.examples.northwind_qna

You: How many orders were shipped to Germany in Q2 1997?
[16:19:52] [orchestrator] START: How many orders were shipped to Germany in Q2 1997?
[16:19:54] [orchestrator] TOOL: handoff({'target_agent': 'data_fetcher'})
[16:19:54] [data_fetcher] START: Previous agent result: [handoff: data_fetcher]...
[16:19:56] [data_fetcher] TOOL: odata_query({'entity': 'Orders', 'filter': "ShipCountry eq 'Germany' and OrderDate ge 1997-04-01T00:00:00Z and OrderDate lt 1997-07-01T00:00:00Z"})
[16:19:57] [data_fetcher] RESULT: {"status": "ok", "count": 20, ...}
[16:19:58] [data_fetcher] TOOL: handoff({'target_agent': 'data_analyst'})
[16:19:58] [data_analyst] TOOL: analyze_data({'code': "from datetime import datetime..."})
[16:19:59] [data_analyst] RESULT: 20 orders in Q2 1997 to Germany. 16 within 3 weeks, 4 longer.

Answer: 20 orders were shipped to Germany in Q2 1997. 16 were delivered within
        3 weeks; 4 took longer.
```

See [`Northwind_test_questions.md`](examples/Northwind_test_questions.md) for test questions.

## Requirements

- Python 3.10+
- See `requirements.txt`

