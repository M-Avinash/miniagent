# agent-core — A Minimal Agent Framework

A small, composable agent framework built from scratch. No agent frameworks, no orchestration libraries — just the primitives that matter.

## Design Philosophy

An agent is a **loop**: the LLM observes, decides, acts, and observes again. Everything else is support structure around that loop. This framework provides the minimal set of primitives that make that loop useful:

1. **Tool calling** — LLM-native function calling with JSON schema, auto-generated from Python type hints
2. **Agent loop** — the observe → think → act → observe cycle with turn limits
3. **Structured output** — Pydantic-validated responses from the LLM
4. **Memory** — conversation history with token-aware truncation
5. **Hooks** — middleware-style interceptors for guardrails, logging, rate limiting
6. **Multi-agent** — agents can call other agents as tools (composition)

### What I deliberately left out

- **Orchestration/graph engines** — for a small framework, the loop *is* the orchestration. Adding a DAG engine adds complexity without proportional value at this scale.
- **RAG / vector stores** — that's a data retrieval concern, not an agent concern. Tools can wrap any retrieval system.
- **Streaming** — the LLM client supports it, but the agent loop is synchronous for simplicity. Streaming can be layered on without changing the core.
- **Persistence** — memory is in-process. A production framework would add serialization, but that's infrastructure, not architecture.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ System     │  │ Tool       │  │ Hook         │  │
│  │ Prompt     │  │ Registry   │  │ Pipeline     │  │
│  └────────────┘  └────────────┘  └──────────────┘  │
│                                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │              Agent Loop (run_async)            │  │
│  │                                                │  │
│  │  for turn in 1..max_turns:                     │  │
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
│  │         Memory (token-aware context)           │  │
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

**1. The `@tool` decorator does the heavy lifting.**
A Python function with type hints and a docstring becomes a fully-specified tool. The framework introspects the signature to build the JSON schema, parses the docstring for parameter descriptions, and handles sync/async execution. This means defining a tool is as simple as writing a function.

**2. Hooks are the extension point, not a plugin system.**
Instead of a complex plugin architecture, hooks are just functions that receive an event and can optionally return overrides. This is the same pattern as middleware in web frameworks — simple, composable, and easy to reason about. A guardrail is a hook. A logger is a hook. A rate limiter is a hook.

**3. Multi-agent via composition, not orchestration.**
`agent.as_tool()` wraps an agent as a callable tool. The parent agent sees it as just another tool — no special orchestration logic needed. This keeps the mental model simple: agents are tools that happen to have their own LLM loop.

**4. Structured output via prompt engineering, not API features.**
Rather than relying on provider-specific `response_format` or `structured_outputs` features, the framework injects the Pydantic schema into the system prompt and parses the JSON from the response. This works with any LLM, including local models via llama.cpp.

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Set your API key (or use a local server)
export OPENAI_API_KEY="sk-..."
# For local:
# export OPENAI_API_KEY="dummy"
# export OPENAI_BASE_URL="http://localhost:8080/v1"

# 3. Run the example agents
python examples/research_agent.py
python examples/guardrails.py
```

## Defining Your Own Agent

```python
from agent_core import Agent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information. Args: query (str): The search query."""
    # ... implement search ...
    return "results..."

@tool
def read_url(url: str) -> str:
    """Fetch and extract text from a URL. Args: url (str): The URL to fetch."""
    # ... implement fetch ...
    return "content..."

agent = Agent(
    name="researcher",
    model="gpt-4o",
    system_prompt="You are a research assistant. Use tools to find accurate information.",
    tools=[search_web, read_url],
    max_turns=10,
)

result = agent.run("What are the latest developments in quantum computing?")
print(result.text)
print(f"Turns: {result.turns}, Tool calls: {len(result.tool_calls)}")
```

### Structured Output

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    summary: str = Field(description="A concise summary")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Sources used")

result = agent.run_structured("What is the capital of France?", schema=Answer)
print(result.structured.summary)       # "Paris"
print(result.structured.confidence)    # 0.99
```

### Hooks (Guardrails, Logging, Rate Limiting)

```python
from agent_core import EventType, HookEvent

def guardrail(agent_name: str, event: HookEvent) -> dict | None:
    if event.type == EventType.AGENT_START:
        if "ignore all previous" in event.data.get("input", "").lower():
            return {"abort": True, "message": "⛔ Blocked."}
    return None

agent = Agent(..., hooks=[guardrail])
```

### Multi-Agent

```python
researcher = Agent(name="researcher", tools=[search_web, read_url])
writer = Agent(name="writer", tools=[researcher.as_tool()])
result = writer.run("Write a blog post about quantum computing")
```

## Project Structure

```
agent_core/
├── __init__.py          # Public API
├── agent.py             # Agent class & core loop
├── llm.py               # LLM client (OpenAI-compatible)
├── tool.py              # @tool decorator & schema introspection
├── memory.py            # Conversation memory & token management
├── hooks.py             # Hook pipeline
├── structured.py        # Structured output (Pydantic)
└── types.py             # Shared types (events, results)
examples/
├── research_agent.py    # Research agent with web tools + multi-agent
└── guardrails.py        # Guardrails, rate limiting, audit logging
```

## Development Approach

### Coding Agent Used: **Claude** (Anthropic's coding agent, via the pi CLI)

### How I Worked

The framework was built iteratively in ~4 hours, working with Claude as a pair programmer:

1. **Scaffold (30 min)** — Defined the core abstractions: what is an Agent, what is a Tool, what is a Hook. Wrote the type system and module structure. Claude helped validate the API surface and catch design issues early.

2. **Tool system (45 min)** — Built the `@tool` decorator with schema introspection. The tricky part was docstring parsing for parameter descriptions. Claude caught the naming conflict between the `Tool` class and the `@tool` decorator, and the edge case where custom tool names need to override the schema's function name.

3. **Agent loop (45 min)** — Implemented the core observe→think→act cycle. Claude helped with the OpenAI tool-calling message format (assistant message with `tool_calls` array, followed by `role: "tool"` messages with `tool_call_id`). Tested against a local llama.cpp server running Qwen3.8-27B.

4. **Structured output (20 min)** — Added Pydantic schema injection into the system prompt and JSON extraction from LLM responses. Claude suggested handling markdown code fences and extra text around the JSON.

5. **Hooks (20 min)** — Added the hook pipeline. Claude helped design the event system (typed events with optional data) and the override mechanism (hooks can return `{"abort": True}` to stop the loop).

6. **Multi-agent (15 min)** — Added `agent.as_tool()` for composition. Claude caught the bug where the internal function name was leaking into the tool schema instead of the custom name.

7. **Examples & validation (30 min)** — Built the research agent and guardrails examples. Ran end-to-end tests against the local LLM server to verify tool calling, structured output, multi-agent delegation, and hook interception all work.

### Validation

- ✅ Tool calling: Agent correctly calls `calculate("15 * 23 + 7")` → gets `352`
- ✅ Structured output: `run_structured()` returns validated Pydantic model
- ✅ Multi-agent: Writer agent delegates to Researcher agent via `as_tool()`
- ✅ Hooks: Guardrail blocks prompt injection, rate limiter caps tool calls
- ✅ Memory: Token-aware truncation keeps context within budget
- ✅ Works with any OpenAI-compatible endpoint (tested with llama.cpp + Qwen3.8-27B)

## Requirements

- Python 3.10+
- An OpenAI-compatible API (OpenAI, Ollama, vLLM, llama.cpp server, etc.)

## License

MIT
