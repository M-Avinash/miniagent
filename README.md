# miniagent

A minimal agent framework. Built with **Pi coding agent** and a self-hosted **Qwen3.8 27B LLM (llama.cpp, RTX 3090)**. The agentic loop lives in `miniagent/agent.py`; everything else is a tool, a hook, or a wrapper. Runs on any **OpenAI-compatible API**. Scales from single agent to multi-agent handoff chains.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│         Agentic Framework with orchestration               │
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

## Layout

```
miniagent/
  __init__.py       — public API exports
  agent.py          — agentic loop (reason → act → observe)
  llm.py            — LLM client (OpenAI-compatible, retry on 429)
  tool.py           — @tool decorator + make_handoff_tool
  hooks.py          — hook pipeline + handoff_hook
  events.py         — EventType, HookEvent, AgentResult
  memory.py         — bounded conversation window
  structured.py     — JSON extraction + validation
  sandbox.py        — run_python (isolated subprocess)
  observe.py        — verbose_log + trace_hook (JSONL audit trail)
  observation.py    — ObservationalMemory (cross-run JSONL)
  orchestrator.py   — HandoffOrchestrator + handoff/abort markers
  graph_memory.py   — GraphMemory + hooks
  examples/
    northwind/      — multi-agent OData QnA chatbot (app, tools, agents, schema)
  tests/
    conftest.py                — pytest config (live marker, import path)
    test_unit.py               — offline unit tests (no network, CI-safe)
    test_live.py               — live LLM + Northwind OData integration tests
    northwind_ground_truth.json — precomputed answers for the ground-truth checks
docs/
  modules.md        — what each module does
  validation.md     — how the framework was tested
```

## Setup & Run

**Step 1:** Clone and install

```bash
git clone https://github.com/M-Avinash/miniagent
cd miniagent
pip install .
```

**Step 2:** Configure your LLM endpoint

Copy `.env.example` to `config.env` (repo root) and edit it:

```bash
cp .env.example config.env
```

Config resolution order: **constructor argument > environment variable > `config.env` > built-in default.** `config.env` is gitignored — it holds your keys and local server URL.

For a **local server** (llama.cpp / Ollama / vLLM) — the tested setup:

```
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8080/v1
MINIAGENT_MODEL=Qwen3.8-27B
MINIAGENT_TOOL_CHOICE=required
```

For **OpenAI** (default — just set the key, no base URL):

```
OPENAI_API_KEY=sk-...
MINIAGENT_MODEL=gpt-5.6-luna
MINIAGENT_TOOL_CHOICE=auto
```

> `MINIAGENT_TOOL_CHOICE=required` is needed for some local models (e.g. Qwen) to reliably call tools; `auto` is the OpenAI default.

**Step 3:** Run the example

```bash
python -m miniagent.examples.northwind.app
```

You're now in the chatbot. Type a question and press Enter. Type `reset` to clear context, `quit` to exit.

**Step 4 (optional):** Run the test suite

```bash
# Offline unit tests (no network, no LLM) — safe for CI:
pytest miniagent/tests/test_unit.py

# Live integration tests (need a reachable LLM + the Northwind OData service):
MINIAGENT_RUN_LIVE=1 pytest miniagent/tests/test_live.py
```

`test_unit.py` covers tools, sandbox, hooks, memory, structured output, and the orchestrator's state handling — all offline. `test_live.py` runs real LLM integration tests plus the Northwind example checked against `northwind_ground_truth.json` (answers precomputed with pure Python, no LLM). Live tests are skipped unless `MINIAGENT_RUN_LIVE=1` is set.

## Build Your Own Agent

An agent is an LLM client, a set of tools, and optional hooks. Tools are plain Python functions; the `@tool` decorator turns type hints and the docstring into a JSON schema.

```python
from miniagent import Agent, LLMClient, tool, verbose_log

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

For multi-agent chains, give each agent a `make_handoff_tool(...)` and run them with `HandoffOrchestrator`.

### Multi-agent patterns

| Pattern | API | Use when |
|---------|-----|----------|
| **Handoff chain** | `HandoffOrchestrator([a, b, c])` | Fixed pipeline: A → B → C |
| **Supervisor** | `make_supervisor_tool(workers)` | One agent dynamically routes to N workers |
| **Agent as tool** | `make_agent_tool(sub_agent)` | Nest agents: parent calls sub-agent like a tool |

To run agents in parallel, use your own `concurrent.futures.ThreadPoolExecutor` — each `Agent.run()` call is independent, so one thread per agent is all you need.

### Observability

- **`verbose_log(label)`** — human-readable terminal log, per-agent labeled.
- **`trace_hook(path, run_id)`** — machine-readable JSONL audit trail. One line per event (LLM call, tool call, result, start/end), timestamped, with `run_id` for correlating multi-agent runs.

### Failure routing

`HandoffOrchestrator` accepts an `on_failure` callback: `(failed_agent, result, question) -> AgentResult | Agent | str | None`. Return a new `Agent` to retry with a different agent, a `str` to retry with a new message, an `AgentResult` to short-circuit, or `None` to accept the abort.

### Structured output

Pass a dataclass or pydantic model as `output_type` to get a typed result instead of free text:

```python
@dataclass
class Answer:
    text: str
    confidence: float

agent = Agent(..., output_type=Answer)
result = agent.run("Is the sky blue?")
result.structured  # Answer(text='Yes', confidence=0.95)
```

## Design Choices

- **Minimal loop, everything else is a hook.** The reason → act → observe loop in `agent.py` stays small. Logging, entity tracking, self-healing and routing are all hooks, so behavior can be changed without touching the core.
- **Layered memory.** Bounded window within a run, observational memory (distilled summaries) across runs, graph memory for records in scope.
- **Code execution instead of reasoning over raw data.** The analyst writes Python that runs in a sandboxed subprocess. More accurate than asking the model to read rows, and errors come back as strings so the agent can self-correct.
- **Shared state instead of passing data through tool arguments.** Fetched datasets live in a dict that agents read directly, avoiding pushing large JSON through the context window.
- **Multi-agent handoffs with single-job agents.** Narrow tools and prompts per agent make routing explicit and each step easier to test.
- **Provider-agnostic.** Any OpenAI-compatible endpoint works.

## Reference

- [Modules](docs/modules.md) — what each file does
- [Northwind example](miniagent/examples/northwind/README.md) — how the three agents fit together
- [Validation approach](docs/validation.md) — how the framework was tested
- [Test questions](miniagent/examples/Northwind_test_questions.md) — single-shot and multi-turn questions with ground truth
