# miniagent — A Lean, Robust Agent Framework

A small agent framework built from scratch with **pi coding agent** on a **local Qwen3 27B** (RTX 3090, llama.cpp). The loop is the orchestration — no graphs, no DAGs, no orchestration engines. Just the primitives that make an agent loop useful.

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

**1. The loop is the orchestration.**
`agent.py` contains the *only* tool-calling loop. The playground, heartbeat, and any future code all call `agent.run()`. No separate loops, no DAGs, no orchestration engines.

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
├── playground.py        # Interactive driving range
├── config.env           # Local config (gitignored)
└── .env.example         # Config template
```

## Development Approach

### Coding Agent Used: **pi coding agent** (local LLM: Qwen3 27B, RTX 3090, llama.cpp)

### How I Worked

The framework was built iteratively, working with pi as a pair programmer on a local LLM (Qwen3 27B on an RTX 3090 via llama.cpp):

1. **Scaffold** — Defined the core abstractions: what is an Agent, what is a Tool, what is a Hook. Wrote the event system and module structure.

2. **LLM client** — Built the OpenAI-compatible client with config precedence (args > env/config.env > defaults). Tested against a local llama.cpp server.

3. **Tool system** — Built the `@tool` decorator with schema introspection. The tricky part was docstring parsing for parameter descriptions and handling the dual-mode decorator (bare `@tool` vs. `@tool(name=..., description=...)`).

4. **Agent loop** — Implemented the unified tool-calling loop. Tools stay available every turn; `tool_choice="required"` on turn 1, `"auto"` after. The model stops naturally when it decides to answer. Crash-safe tool execution, bounded by `max_tool_calls`.

5. **Structured output** — JSON extraction from messy LLM text (fenced, chatter, nested, garbage) + optional field validation. No Pydantic.

6. **Hooks** — Middleware pattern. Hooks can observe, modify, or abort. Each hook is wrapped in try/except — a buggy hook never kills the agent.

7. **Heartbeat** — Interval runner with optional maker-checker. Not a new loop — just a while-loop with a sleep that calls `agent.run()`.

8. **Sandbox** — `run_python` tool. Subprocess with timeout. Errors and timeouts come back as strings, never as exceptions that kill the agent.

9. **Observability** — `verbose_log` hook. Per-agent labeled event printing. Optional debug level and JSONL-to-file mode.

10. **Observational memory** — Distilled observations (JSONL store, capped, recall into system prompt). No embeddings, no RAG.

### Validation

- ✅ Tool calling: Agent correctly calls tools, handles bad JSON, unknown tools, and tool exceptions
- ✅ Structured output: `parse_structured()` handles clean JSON, fenced JSON, chatter, nested, and garbage
- ✅ Heartbeat: Bounded runs, `run_forever` with `max_beats`, maker-checker
- ✅ Sandbox: Basic execution, runtime errors, syntax errors, timeouts, non-zero exit
- ✅ Observability: `verbose_log` prints every event with per-agent labeling
- ✅ Observational memory: Distills runs into observations, recalls into system prompt
- ✅ Works with any OpenAI-compatible endpoint (tested with llama.cpp + Qwen3 27B)

## Requirements

- Python 3.10+
- An OpenAI-compatible API (OpenAI, Ollama, vLLM, llama.cpp server, etc.)

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
