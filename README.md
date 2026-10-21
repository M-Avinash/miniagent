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
  observe.py        — verbose_log hook
  observation.py    — ObservationalMemory (cross-run JSONL)
  orchestrator.py   — HandoffOrchestrator + handoff/abort markers
  graph_memory.py   — GraphMemory + hooks
  examples/
    northwind/      — multi-agent OData QnA chatbot (app, tools, agents, schema)
  tests/
    test_framework.py          — 121 checks (unit + live LLM integration)
    northwind_ground_truth.json
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
python miniagent/tests/test_framework.py
```

121 checks: unit tests (tools, sandbox, hooks, memory, structured output, orchestrator) plus live LLM integration tests and the Northwind example against ground truth. Requires a reachable LLM endpoint (see Step 2).

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
| **Parallel fan-out** | `fan_out([a, b, c], [msg1, msg2, msg3])` | Run N agents in parallel, collect results |

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
