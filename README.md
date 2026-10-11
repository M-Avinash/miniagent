# miniagent — A Lean, Robust Agent Framework

A small agent framework built from scratch with **pi coding agent** and a **self-hosted Qwen3 27B** (llama.cpp on an RTX 3090). It implements the full six-layer agent stack — **model core, memory, tools, planner, orchestration runtime, and observability** — in a handful of modules, with a single **ReAct loop** (reason → act → observe) as the spine. It runs on any OpenAI-compatible API and scales from a single agent to a multi-agent **handoff** chain.

> **Live example:** a multi-agent QnA chatbot over the SAP Northwind OData service — ask questions in natural language, get answers computed from live data. See [Northwind QnA](#northwind-qna-multi-agent-example).

## What you get

`miniagent` covers the standard agent architecture layers, each as a small, focused module:

1. **Model core** — a provider-agnostic `LLMClient` for any OpenAI-compatible endpoint, with retry-on-rate-limit and reasoning-model support.
2. **Reasoning loop (ReAct)** — the observe → think → act → observe cycle, with parallel tool calls, a bounded termination policy (`max_tool_calls`), and a single source of truth in `agent.py`.
3. **Typed tools** — LLM-native function calling with JSON schema auto-generated from Python type hints + docstrings. Includes a **code-execution sandbox** (subprocess, timeout, errors as strings).
4. **Memory** — three complementary types: **working** (the conversation window), **episodic** (long-term observational memory, JSONL store), and **procedural/semantic** (a graph of entity keys for relational schemas).
5. **Structured output** — JSON extraction + field validation, with no Pydantic dependency.
6. **Guardrails & hooks** — middleware-style interceptors at every event for logging, guardrails, abort, and self-healing.
7. **Orchestration runtime** — a **handoff** mechanism (handoff tool + hook + `HandoffOrchestrator`) for supervisor-worker / multi-agent chains, plus a **proactive heartbeat** (OpenClaw-style awakener with `HEARTBEAT_OK` no-ops and an `on_alert` callback, and an optional maker-checker mode).
8. **Observability** — a `verbose_log` hook that emits labeled event spans per agent.

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

### Key design decisions

**1. The ReAct loop is the spine — one loop, one source of truth.**
`agent.py` implements the standard **ReAct** pattern: the model reasons, calls a tool, observes the result, and repeats until it produces a final answer. It is the *only* tool-calling loop in the framework — the heartbeat, orchestrator, and any future code all call `agent.run()`. The loop carries the termination policy (bounded by `max_tool_calls`) and fires hooks at every step, so crash-safety, guardrails, and observability all hang off one place. When you need *multiple* agents, the handoff orchestrator is just a loop over `agent.run()` calls (see below).

**2. The `@tool` decorator does the heavy lifting.**
A Python function with type hints and a docstring becomes a fully-specified tool. The framework introspects the signature to build the JSON schema, parses the docstring for parameter descriptions, and handles sync/async execution. Defining a tool is as simple as writing a function.

**3. Hooks are the guardrail layer.**
Hooks are functions that receive an event and can observe, mutate, or short-circuit it — the same middleware pattern as web frameworks. This is how you attach **guardrails**, observability, abort conditions, and self-healing without touching the loop. A logger is a hook. A guardrail is a hook. An abort condition is a hook.

**4. Structured output via prompt engineering, not API features.**
Rather than relying on provider-specific `response_format` or `structured_outputs` features, the framework injects the JSON schema into the system prompt and parses the JSON from the response. This works with any LLM, including local models via llama.cpp.

**5. Robustness at every layer.**
- **Tools** are crash-safe: bad JSON, unknown tools, and tool exceptions return `[error: ...]` strings, never crash the agent.
- **The loop** is bounded by `max_tool_calls` (default 10) — a termination policy that returns an aborted result if the model doesn't converge.
- **The sandbox** runs code in a subprocess with a timeout — a hang or crash only kills the subprocess, not the agent.
- **Hooks** are wrapped in try/except — a buggy guardrail never kills the agent.
- **The orchestrator** is bounded by `max_handoffs` — a handoff loop can never run forever.
- **The LLM client** retries on 429 rate-limit errors with exponential backoff, so bursty multi-agent runs survive provider throttling.

### A provider-agnostic LLM client

`LLMClient` talks to any OpenAI-compatible endpoint — OpenAI, Ollama, vLLM, llama.cpp, Groq, Together. Switch providers by editing one config file; no code changes. It handles the provider quirks that usually trip people up:

- **Tool choice** — `auto` for OpenAI, `required` for local models that need a nudge to call tools.
- **Reasoning models** — set `reasoning_effort=none` to use function tools with reasoning models (e.g. `gpt-5.6-luna`) via Chat Completions.
- **Token caps** — set `max_tokens` to keep local models from generating until the context limit.
- **Retry on rate limit** — automatic exponential backoff (2s, 5s, 15s) on 429 errors, capped and configurable.

## Quick Start

`miniagent` works with any OpenAI-compatible API. Configure it once via `config.env` (or environment variables), then run.

**OpenAI (most common):**

```bash
cat > miniagent/config.env <<EOF
OPENAI_API_KEY=sk-...
MINIAGENT_MODEL=gpt-5.6-luna
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_TOOL_CHOICE=auto
MINIAGENT_REASONING_EFFORT=none
EOF
```

**Local model (llama.cpp / Ollama / vLLM):**

```bash
cat > miniagent/config.env <<EOF
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8080/v1
MINIAGENT_MODEL=qwen3
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_MAX_TOKENS=1024
MINIAGENT_TOOL_CHOICE=required
EOF
```

Verify the setup with a one-line smoke test:

```bash
python3 -c "from miniagent import LLMClient; print(LLMClient().chat([{'role':'user','content':'Reply OK'}]).choices[0].message.content)"
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

### Heartbeat (proactive awakener + maker-checker)

A `HeartbeatAgent` wakes an agent on an interval — the OpenClaw-style pattern that turns a reactive chatbot into a **proactive agent** that works while you are away. It is not a new loop: it is a plain while-loop with a sleep that keeps calling the existing `agent.run()`.

**Proactive awakener (default).** The heartbeat sends a "check for pending work" prompt. The agent runs its tools, and if nothing needs attention it replies `HEARTBEAT_OK` — a no-op that is suppressed. Only when the agent surfaces something does an alert fire. Wire up delivery with the `on_alert` callback:

```python
from miniagent import Agent, HeartbeatAgent, HEARTBEAT_OK

monitor = Agent(
    "On each heartbeat, check recent orders for delays. If any, alert. "
    "Otherwise reply with exactly HEARTBEAT_OK.",
    tools=[odata_query, analyze_data],
)

def on_alert(result):
    print("ALERT:", result.text)

hb = HeartbeatAgent(monitor, interval=300, on_alert=on_alert)
hb.run_forever()  # or hb.run(n_times=5) for a bounded demo
```

**Maker-checker (optional).** Pass a `checker` agent and the heartbeat keeps beating until the checker reports the task is done — a convergence loop rather than an awakener.

See [`northwind_heartbeat.py`](northwind_heartbeat.py) for a runnable showcase: a monitor that queries the Northwind data for delayed shipments and freight spikes, no-ops with `HEARTBEAT_OK` when all is normal, and writes alerts to `alerts.jsonl`. (Northwind is just a convenient, self-contained data source for the demo — a real deployment would point the monitor at something that changes over time.)

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

### The three memory types at a glance

`miniagent` ships the three standard agent memory types — **working**, **episodic**, and **procedural/semantic** — as separate modules. Each answers a different "remember" question, and they don't overlap:

| Memory type | Module | Scope | Persistent? | What it remembers |
|--------|--------|-------|-------------|-------------------|
| **Working** (conversation window) | `memory.py` | One run | No | The current conversation (messages sent to the LLM) |
| **Episodic** (observational memory) | `observation.py` | Across runs | Yes (JSONL) | Distilled summaries of past runs (long-term recall) |
| **Procedural/semantic** (entity graph) | `graph_memory.py` | Across turns | No (RAM) | Exact entity keys + navigation path (relational focus) |

- **Working memory** is the context for the *current* question — wiped at the start of each `run()`.
- **Episodic memory** is the agent's journal — it survives restarts and lets the agent recall what it did before.
- **Procedural/semantic memory** tracks *which entities* you're currently focused on, so multi-turn drill-down ("which of *those* were delayed?") doesn't lose the thread.

Use them together for a full-memory agent: working memory for the present, episodic memory for the past, and the graph for relational navigation.

## Handoff Orchestrator

When one agent isn't enough, `HandoffOrchestrator` implements the **handoff** pattern (a.k.a. supervisor-worker): a chain of specialized agents where each agent delegates to the next by setting `event.data["handoff"]`. The orchestrator follows the chain until an agent returns a result with no handoff (or `max_handoffs` is hit). This is the same mechanism the OpenAI Agents SDK and CrewAI use for multi-agent routing.

There are two ways to set a handoff:

### Option 1: Handoff tool (recommended)

Use the `make_handoff_tool()` and `handoff_hook()` from the framework. The agent calls a `handoff` tool, and the hook reads the tool call and sets the handoff. This is the cleanest, most prod-grade approach.

```python
from miniagent import Agent, HandoffOrchestrator, make_handoff_tool, handoff_hook

planner = Agent(
    "You plan. Then hand off to the analyst.",
    tools=[make_handoff_tool(["analyst"])],
    hooks=[handoff_hook],
    name="planner",
)
analyst = Agent("You analyze.", name="analyst")

orch = HandoffOrchestrator(
    [planner, analyst],
    max_handoffs=5,
    agent_names={"planner": planner, "analyst": analyst},
)
result = orch.run("Analyze the sales data for Q2")
```

The `make_handoff_tool(["analyst"])` creates a tool that the LLM can call with `target_agent='analyst'`. The `handoff_hook` reads the tool call and sets `event.data["handoff"] = 'analyst'`, which the orchestrator follows.

### Option 2: Custom hook

You can also set the handoff in a custom hook (e.g. on `AGENT_END`, inspecting the final text). This is more flexible but less structured.

```python
from miniagent import Agent, HandoffOrchestrator, EventType

def handoff_to_analyst(event):
    if event.type == EventType.AGENT_END:
        if "HANDOFF" in event.data.get("final_text", ""):
            event.data["handoff"] = "analyst"
    return event

planner = Agent("You plan. Then hand off.", hooks=[handoff_to_analyst], name="planner")
analyst = Agent("You analyze.", name="analyst")

orch = HandoffOrchestrator([planner, analyst], max_handoffs=5)
result = orch.run("Analyze the sales data for Q2")
```

**How handoffs work:** an agent sets `event.data["handoff"]` to the name of the next agent (via a tool call or a hook). The orchestrator looks that name up and runs the next agent — a simple loop over `agent.run()` calls, easy to follow and extend.

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

The chatbot is **multi-turn**: `shared_state` and `GraphMemory` persist across questions, so the agent can reference prior context ("which of *those* were delayed?"). Type `reset` to clear the context.

## Semantic Graph Memory

For relational schemas (OData, SQL, etc.), flat chat history causes **context drift** and **key hallucination** in multi-turn conversations — the LLM has to guess which entity is "active" from the transcript. `GraphMemory` solves this by tracking exact entity keys and the navigation path, then injecting them into the prompt.

It's a Python dict of entity keys plus a path list: deterministic, zero-latency, and compressible (the prompt block stays ~3 lines regardless of how many records you've seen).

```python
from miniagent import GraphMemory, graph_memory_hook, graph_injection_hook

graph = GraphMemory()
agent = Agent(
    system_prompt,
    tools=[odata_query],
    hooks=[
        graph_memory_hook(graph, {"Order": "OrderID", "Customer": "CustomerID"}),
        graph_injection_hook(graph),
        verbose_log("agent"),
    ],
)
```

After each `odata_query`, the `graph_memory_hook` extracts the primary keys and adds them to the graph. Before each LLM call, the `graph_injection_hook` prepends a compressed context block to the system prompt:

```
[SEMANTIC GRAPH MEMORY ACTIVE]
Current Data Focus Path: Customer('ALFKI') -> Order('10248')
Available Entity Keys in Memory: {'Customer': ['ALFKI'], 'Order': ['10248']}
```

The LLM now has explicit entity keys — no guessing, no drift.

## Project Structure

```
miniagent/
├── __init__.py          # Public API
├── agent.py             # Agent class & core loop (one source of truth)
├── llm.py               # LLM client (any OpenAI-compatible API, retry on 429)
├── tool.py              # @tool decorator & schema introspection
├── memory.py            # Bounded conversation window
├── hooks.py             # Hook pipeline
├── structured.py        # Structured output (JSON extraction + validation)
├── events.py            # EventType, HookEvent, AgentResult
├── heartbeat.py         # HeartbeatAgent (proactive awakener + maker-checker)
├── sandbox.py           # run_python tool (subprocess execution)
├── observe.py           # verbose_log hook
├── observation.py       # ObservationalMemory (long-term distilled JSONL store)
├── orchestrator.py      # HandoffOrchestrator + make_handoff_tool + handoff_hook
├── graph_memory.py      # GraphMemory + hooks (semantic graph memory for relational schemas)
├── northwind_qna.py     # Example: multi-agent OData QnA chatbot
├── northwind_heartbeat.py  # Showcase: HeartbeatAgent demonstrated on Northwind data
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

Config resolution order: **constructor argument > environment / `config.env` > built-in default.** Real environment variables always win over `config.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `dummy` | API key (any value for local servers) |
| `OPENAI_BASE_URL` | — | Base URL. Leave unset for OpenAI; set for local servers (e.g. `http://localhost:8080/v1`) |
| `MINIAGENT_MODEL` | `gpt-4o` | Model name (e.g. `gpt-5.6-luna`, `gpt-5.2`, `qwen3`) |
| `MINIAGENT_TEMPERATURE` | `0.7` | Sampling temperature |
| `MINIAGENT_MAX_TOKENS` | — | Max tokens per LLM call (important for local models!) |
| `MINIAGENT_TOOL_CHOICE` | `auto` | Tool choice: `auto` (OpenAI) or `required` (some local models like Qwen) |
| `MINIAGENT_REASONING_EFFORT` | — | Reasoning effort. Set to `none` for reasoning models (e.g. `gpt-5.6-luna`) to enable function tools via Chat Completions |
| `MINIAGENT_MAX_RETRIES` | `3` | Retries on a 429 rate-limit error, with exponential backoff (2s, 5s, 15s) |

**Notes:**
- **Local models:** always set `MINIAGENT_MAX_TOKENS`. Without a cap, local models (especially "aggressive" fine-tunes) can generate until the context limit, causing multi-minute hangs.
- **Reasoning models:** models like `gpt-5.6-luna` and the o-series need `MINIAGENT_REASONING_EFFORT=none` to use function tools via the Chat Completions API (otherwise OpenAI returns a 400).
- **Rate limits:** the client automatically retries on 429 errors with exponential backoff, so bursty multi-agent runs survive OpenAI's throttling. Increase `MINIAGENT_MAX_RETRIES` for longer runs, or set it to `0` to disable.

## License

MIT
