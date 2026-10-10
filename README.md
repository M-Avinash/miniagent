# miniagent — A Lean, Robust Agent Framework

A small agent framework built from scratch with **pi coding agent** on a **local Qwen3 27B** (RTX 3090, llama.cpp). The loop is the orchestration — no graphs, no DAGs, no orchestration engines. Just the primitives that make an agent loop useful, plus a handoff orchestrator for when you genuinely need more than one agent.

> **Live example:** a multi-agent QnA chatbot over the SAP Northwind OData service — ask questions in natural language, get answers computed from live data. See [Northwind QnA](#northwind-qna-multi-agent-example).

## Design Philosophy

An agent is a **loop**: the LLM observes, decides, acts, and observes again. Everything else is support structure around that loop. `miniagent` provides the minimal set of primitives that make that loop robust and useful:

1. **Tool calling** — LLM-native function calling with JSON schema, auto-generated from Python type hints + docstrings
2. **Agent loop** — the unified observe → think → act → observe cycle (one source of truth in `agent.py`)
3. **Structured output** — JSON extraction + field validation (no Pydantic)
4. **Memory** — bounded conversation list (no token counting)
5. **Hooks** — middleware-style interceptors for logging, guardrails, abort
6. **Heartbeat** — run an agent on an interval (optional maker-checker)
7. **Sandbox** — execute Python code in a subprocess (timeout, errors as strings)
8. **Observability** — `verbose_log` hook (per-agent labeled event printing)
9. **Observational memory** — distilled observations (JSONL store, capped, recall into system prompt)
10. **Handoff orchestrator** — run multiple agents, each handing off to the next (no graph, no DAG)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ System     │  │ Tools      │  │ Hook         │  │
│  │ Prompt     │  │ (list)     │  │ Pipeline     │  │
│  └────────────┘  └────────────┘  └──────────────┘  │
│                                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │              Agent Loop (run)                  │  │
│  │                                                │  │
│  │  for turn in 1..max_tool_calls+1:              │  │
│  │    messages → LLM.chat(tools) → response       │  │
│  │    if response.tool_calls:                     │  │
│  │      for each tool_call:                       │  │
│  │        hooks → execute tool → append result    │  │
│  │      continue                                  │  │
│  │    else:                                       │  │
│  │      final_text = response.content; break      │  │
│  └────────────────────────────────────────────────┘  │
│                                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │         Memory (bounded list)                  │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    LLM Client       │  OpenAI-compatible API
│  (swappable)        │  (OpenAI, Ollama, vLLM, llama.cpp, ...)
└─────────────────────┘
```

### Key design decisions

**1. The loop is the orchestration — for a single agent.**
`agent.py` contains the *only* tool-calling loop. The playground, heartbeat, and any future code all call `agent.run()`. No separate loops, no DAGs, no orchestration engines. When you need *multiple* agents, you don't reach for a graph — you reach for the handoff orchestrator, which is just a loop over `agent.run()` calls (see below).

**2. The `@tool` decorator does the heavy lifting.**
A Python function with type hints and a docstring becomes a fully-specified tool. The framework introspects the signature to build the JSON schema, parses the docstring for parameter descriptions, and handles sync/async execution. Defining a tool is as simple as writing a function.

**3. Hooks are the extension point, not a plugin system.**
Instead of a complex plugin architecture, hooks are just functions that receive an event and can optionally return overrides. This is the same pattern as middleware in web frameworks — simple, composable, and easy to reason about. A logger is a hook. A guardrail is a hook. An abort condition is a hook.

**4. Structured output via prompt engineering, not API features.**
Rather than relying on provider-specific `response_format` or `structured_outputs` features, the framework injects the JSON schema into the system prompt and parses the JSON from the response. This works with any LLM, including local models via llama.cpp.

**5. Robustness over features.**
- Tool execution is crash-safe: bad JSON, unknown tools, and tool exceptions return `[error: ...]` strings, never crash the agent.
- The agent is bounded by `max_tool_calls` (default 10) — if the model doesn't answer by then, it returns an aborted result.
- The sandbox runs code in a subprocess with a timeout — a hang or crash only kills the subprocess, not the agent.
- Hooks are wrapped in try/except — a buggy hook never kills the agent.
- The orchestrator is bounded by `max_handoffs` — a handoff loop can never run forever.

## Quick Start

```bash
# 1. Configure (or use environment variables)
cat > miniagent/config.env <<EOF
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8080/v1
MINIAGENT_MODEL=qwen3
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_MAX_TOKENS=1024
MINIAGENT_TOOL_CHOICE=required
EOF

# 2. Run the playground
python3 -u miniagent/playground.py
```

## Defining Your Own Agent

```python
from miniagent import Agent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information.
    Args:
        query (str): The search query.
    """
    # ... implement search ...
    return "results..."

agent = Agent(
    "You are a research assistant. Use tools to find accurate information.",
    tools=[search_web],
    max_tool_calls=10,
)

result = agent.run("What are the latest developments in quantum computing?")
print(result.text)
print(f"Turns: {result.turns}, Tool calls: {len(result.tool_calls)}")
```

### Structured Output

```python
from miniagent import Agent, parse_structured

agent = Agent("You are a helpful assistant.")
result = agent.run("What is the capital of France? Respond with JSON: {\"answer\": \"...\"}")
parsed = parse_structured(result.text, {"answer": str})
print(parsed["answer"])  # "Paris"
```

### Hooks (Logging, Guardrails, Abort)

```python
from miniagent import Agent, EventType, HookEvent

def guardrail(event):
    if event.type == EventType.AGENT_START:
        if "ignore all previous" in event.data.get("user_message", "").lower():
            event.data["abort"] = True
    return event

agent = Agent("...", hooks=[guardrail])
```

### Heartbeat

```python
from miniagent import Agent, HeartbeatAgent

maker = Agent("You are a code improver.")
hb = HeartbeatAgent(maker, "Improve the code in file.py", interval=30)
hb.run(n_times=5)  # or hb.run_forever()
```

### Sandbox

```python
from miniagent import Agent, run_python

agent = Agent("You are a math assistant. Use run_python to compute.", tools=[run_python])
result = agent.run("What is the sum of the first 100 even numbers?")
print(result.text)
```

### Observability

```python
from miniagent import Agent, verbose_log

agent = Agent("...", hooks=[verbose_log("my-agent")])
result = agent.run("What is the weather in Paris?")
# Prints: [12:34:56] [my-agent] START: What is the weather in Paris?
#         [12:34:56] [my-agent] LLM call (turn 1, 2 messages)
#         [12:34:57] [my-agent] LLM response (turn 1): '' + 1 tool call(s)
#         [12:34:57] [my-agent] TOOL: get_weather({'city': 'Paris'})
#         ...
```

### Observational Memory

```python
from miniagent import Agent, ObservationalMemory

om = ObservationalMemory("my-agent", path="observations.jsonl", max_entries=50)
agent = Agent("...", hooks=[om.hook])
result = agent.run("What is 2+2?")

# On next run:
system_prompt = base_prompt + om.recall_prompt()
agent2 = Agent(system_prompt, ...)
```

## Handoff Orchestrator

When one agent isn't enough, `HandoffOrchestrator` runs a chain of agents. Each agent can hand off to the next by setting `event.data["handoff"]` in an `AGENT_END` hook. The orchestrator follows the chain until an agent returns a result with no handoff (or `max_handoffs` is hit).

```python
from miniagent import Agent, HandoffOrchestrator, EventType

def handoff_to_analyst(event):
    if event.type == EventType.AGENT_END:
        event.data["handoff"] = "analyst"   # hand off to the agent named "analyst"
    return event

planner = Agent("You plan. Then hand off.", hooks=[handoff_to_analyst], name="planner")
analyst = Agent("You analyze.", name="analyst")

orch = HandoffOrchestrator(
    [planner, analyst],
    max_handoffs=5,
    agent_names={"planner": planner, "analyst": analyst},
)
result = orch.run("Analyze the sales data for Q2")
```

**How handoffs work:** an agent's hook inspects the final text (or any condition) and sets `event.data["handoff"]` to the name of the next agent. The orchestrator looks that name up in `agent_names` and runs it. No graph, no DAG — just a loop over `agent.run()`.

**Sharing data between agents:** don't pass large data through tool-call arguments (local models truncate long JSON). Use a shared state object — the previous agent stores the result, the next agent reads it via a tool. This is what the Northwind example does.

## Northwind QnA (Multi-Agent Example)

A live, working example: a terminal chatbot that answers questions about the SAP Northwind OData service using a three-agent handoff chain.

```
Orchestrator ──handoff──▶ Data Fetcher ──handoff──▶ Data Analyst
   (understands)            (builds OData query)        (runs Python on the data)
```

- **Orchestrator** — reads the question, hands off to the data fetcher.
- **Data Fetcher** — builds and runs the OData query, stores the full result in shared state, hands off to the analyst.
- **Data Analyst** — runs Python against the shared data (via the `analyze_data` tool) and returns the final answer.

```bash
cd miniagent
python -m miniagent.northwind_qna

You: How many orders were shipped to Germany in Q2 1997, and how many were
     delivered within 3 weeks of the order date?
Answer: 20 orders were shipped to Germany in Q2 1997. 16 were delivered within
        3 weeks; 4 took longer.
```

Every step streams to the terminal (via `verbose_log`), labeled by agent. See [`NORTHWIND_TEST_QUESTIONS.md`](NORTHWIND_TEST_QUESTIONS.md) for 10 questions that stress-test joins, date math, and grouping.

## Project Structure

```
miniagent/
├── __init__.py          # Public API
├── agent.py             # Agent class & core loop (one source of truth)
├── llm.py               # LLM client (OpenAI-compatible)
├── tool.py              # @tool decorator & schema introspection
├── memory.py            # Bounded conversation list
├── hooks.py             # Hook pipeline
├── structured.py        # Structured output (JSON extraction + validation)
├── events.py            # EventType, HookEvent, AgentResult
├── heartbeat.py         # HeartbeatAgent (interval runner + maker-checker)
├── sandbox.py           # run_python tool (subprocess execution)
├── observe.py           # verbose_log hook
├── observation.py       # ObservationalMemory (distilled JSONL store)
├── orchestrator.py      # HandoffOrchestrator (multi-agent handoff loop)
├── playground.py        # Interactive driving range
├── northwind_qna.py     # Example: multi-agent OData QnA chatbot
├── NORTHWIND_TEST_QUESTIONS.md  # Test questions for the Northwind example
├── config.env           # Local config (gitignored)
└── .env.example         # Config template
```

## Requirements

- Python 3.10+
- An OpenAI-compatible API (OpenAI, Ollama, vLLM, llama.cpp server, etc.)
- `openai` (Python package)
- `requests` (only for the Northwind OData example)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `dummy` | API key (any value for local servers) |
| `OPENAI_BASE_URL` | — | Base URL (e.g., `http://localhost:8080/v1`) |
| `MINIAGENT_MODEL` | `gpt-4o` | Model name |
| `MINIAGENT_TEMPERATURE` | `0.7` | Sampling temperature |
| `MINIAGENT_MAX_TOKENS` | — | Max tokens per LLM call (important for local models!) |
| `MINIAGENT_TOOL_CHOICE` | `auto` | Tool choice (`required` for local Qwen, `auto` for OpenAI) |

**Note:** Always set `MINIAGENT_MAX_TOKENS` for local models. Without a cap, local models (especially "aggressive" fine-tunes) can generate until the context limit, causing multi-minute hangs.

## License

MIT
