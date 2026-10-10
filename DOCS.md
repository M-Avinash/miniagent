# agent-core — A Minimal Agent Framework

> A small, composable agent framework built from scratch. No agent frameworks,
> no orchestration libraries — just the primitives that matter.
>
> This document merges the framework's README (design + usage) with the
> block-by-block code walkthrough (implementation, in dependency order).

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Usage](#4-usage)
5. [Project Structure](#5-project-structure)
6. [The Code Walkthrough](#6-the-code-walkthrough)
   - [6.0 The Big Picture](#60-the-big-picture)
   - [6.1 `types.py` — the shared vocabulary](#61-typespy--the-shared-vocabulary)
   - [6.2 `llm.py` — talking to the model](#62-llmpy--talking-to-the-model)
   - [6.3 `tool.py` — turning functions into tools](#63-toolpy--turning-functions-into-tools)
   - [6.4 `hooks.py` — the extension point](#64-hookspy--the-extension-point)
   - [6.5 `memory.py` — remembering the conversation](#65-memorypy--remembering-the-conversation)
   - [6.6 `structured.py` — getting typed data back](#66-structuredpy--getting-typed-data-back)
   - [6.7 `agent.py` — the loop (the core)](#67-agentpy--the-loop-the-core)
   - [6.8 `__init__.py` — the public API](#68-initpy--the-public-api)
   - [6.9 How it fits](#69-how-it-fits)
7. [Development Approach](#7-development-approach)
8. [Requirements & License](#8-requirements--license)

---

## 1. Design Philosophy

An agent is a **loop**: the LLM observes, decides, acts, and observes again.
Everything else is support structure around that loop. This framework provides
the minimal set of primitives that make that loop useful:

1. **Tool calling** — LLM-native function calling with JSON schema, auto-generated from Python type hints
2. **Agent loop** — the observe → think → act → observe cycle with turn limits
3. **Structured output** — Pydantic-validated responses from the LLM
4. **Memory** — conversation history with token-aware truncation
5. **Hooks** — middleware-style interceptors for guardrails, logging, rate limiting
6. **Multi-agent** — agents can call other agents as tools (composition)

### What I deliberately left out

- **Orchestration/graph engines** — for a small framework, the loop *is* the
  orchestration. Adding a DAG engine adds complexity without proportional value
  at this scale.
- **RAG / vector stores** — that's a data retrieval concern, not an agent
  concern. Tools can wrap any retrieval system.
- **Streaming** — the LLM client supports it, but the agent loop is synchronous
  for simplicity. Streaming can be layered on without changing the core.
- **Persistence** — memory is in-process. A production framework would add
  serialization, but that's infrastructure, not architecture.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Agent                          │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐ │
│  │ System     │  │ Tool       │  │ Hook           │ │
│  │ Prompt     │  │ Registry   │  │ Pipeline       │ │
│  └────────────┘  └────────────┘  └────────────────┘ │
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
A Python function with type hints and a docstring becomes a fully-specified
tool. The framework introspects the signature to build the JSON schema, parses
the docstring for parameter descriptions, and handles sync/async execution.
This means defining a tool is as simple as writing a function.

**2. Hooks are the extension point, not a plugin system.**
Instead of a complex plugin architecture, hooks are just functions that receive
an event and can optionally return overrides. This is the same pattern as
middleware in web frameworks — simple, composable, and easy to reason about.
A guardrail is a hook. A logger is a hook. A rate limiter is a hook.

**3. Multi-agent via composition, not orchestration.**
`agent.as_tool()` wraps an agent as a callable tool. The parent agent sees it
as just another tool — no special orchestration logic needed. This keeps the
mental model simple: agents are tools that happen to have their own LLM loop.

**4. Structured output via prompt engineering, not API features.**
Rather than relying on provider-specific `response_format` or
`structured_outputs` features, the framework injects the Pydantic schema into
the system prompt and parses the JSON from the response. This works with any
LLM, including local models via llama.cpp.

---

## 3. Quick Start

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

---

## 4. Usage

### Defining Your Own Agent

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

---

## 5. Project Structure

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

---

## 6. The Code Walkthrough

> The framework is explained in **dependency order** — each file only depends
> on files before it. Each block shows the code and explains *why* it exists
> and how it fits.

### Roadmap

| # | File | What it is | Blocks |
|---|------|-----------|--------|
| 0 | The Big Picture | The mental model before any code | 1 |
| 1 | `types.py` | Shared vocabulary (events, results) | 3 |
| 2 | `llm.py` | Talking to the model | 3 |
| 3 | `tool.py` | Turning functions into tools | 6 |
| 4 | `hooks.py` | The extension point | 2 |
| 5 | `memory.py` | Remembering the conversation | 5 |
| 6 | `structured.py` | Getting typed data back | 3 |
| 7 | `agent.py` | **The loop (the core)** | 9 |
| 8 | `__init__.py` | The public API | 1 |
| 9 | How it fits | The wiring diagram | 1 |

**Total: ~34 blocks.**

---

### 6.0 The Big Picture

#### Block 0.1 — What an agent actually is

An agent is just a **loop**:

```
user says something
   ↓
send to LLM
   ↓
LLM says either:
   (a) "here's my answer"        → done
   (b) "call this tool with X"   → run the tool, feed result back to LLM, repeat
```

Every file in the framework exists to support that loop. Nothing more.

- The **LLM client** is how we talk to the model.
- **Tools** are the actions the model can trigger.
- **Memory** is how we remember the conversation between turns.
- **Hooks** are how you intercept the loop (guardrails, logging, rate limits).
- **Structured output** is how you get typed data back instead of free text.

---

### 6.1 `types.py` — the shared vocabulary

This is the "common language" the rest of the code shares. No logic here —
just data shapes.

#### Block 1.1 — `EventType`

```python
class EventType(str, Enum):
    AGENT_START = "agent_start"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_END = "agent_end"
    ERROR = "error"
```

**Why it exists:** The agent loop has several distinct moments (starting,
calling the LLM, getting a response, calling a tool, getting a tool result,
finishing). Hooks need to know *which moment* they're reacting to. This enum
names those moments.

**Why `str` enum:** `EventType.TOOL_CALL == "tool_call"` is `True`. Easy to
compare, print, and log.

#### Block 1.2 — `HookEvent`

```python
@dataclass
class HookEvent:
    type: EventType
    turn: int = 0
    # LLM-related
    messages: list[dict[str, Any]] | None = None
    response: Any = None
    # Tool-related
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    # Error-related
    error: Exception | None = None
    # Arbitrary extra data
    data: dict[str, Any] = field(default_factory=dict)
```

**Why it exists:** This is the *payload* handed to hooks. One object that
carries everything a hook might want about the current moment.

- `type` — which moment (required, no default)
- `turn` — which loop iteration
- `messages` / `response` — set on LLM events
- `tool_name` / `tool_args` / `tool_result` — set on tool events
- `error` — set on error events
- `data` — catch-all dict (e.g. the user's input on `AGENT_START`)

Most fields are `None` because a given event only fills in the relevant ones.
`@dataclass` auto-generates the constructor — no boilerplate.

#### Block 1.3 — `AgentResult`

```python
@dataclass
class AgentResult:
    text: str
    turns: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    structured: Any = None
    usage: dict[str, int] = field(default_factory=dict)
```

**Why it exists:** What `agent.run()` hands back to *you*:

- `text` — the final answer
- `turns` — how many LLM round-trips it took
- `tool_calls` — a log of every tool call (name, args, result) — great for debugging
- `structured` — only filled if you used `run_structured()`
- `usage` — token counts

**Gotcha:** `field(default_factory=list)` instead of `= []`. A mutable default
like `= []` would make every instance share the *same* list. `default_factory`
gives each instance its own fresh list.

---

### 6.2 `llm.py` — talking to the model

The only file that touches the outside world. Everything else is pure Python.

#### Block 2.1 — Constructor

```python
class LLMClient:
    def __init__(self, model="gpt-4o", api_key=None, base_url=None,
                 temperature=0.7, max_tokens=None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key, **kwargs}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
```

**Why it exists:** Wraps the `openai` package so the rest of the framework
never knows about OpenAI's specifics.

**The key trick — `base_url`:** The OpenAI client is *protocol-based*, not
company-based. Point it at a different `base_url` and it talks to whatever
speaks the OpenAI format — Ollama, vLLM, llama.cpp, Groq, Together. That's how
one client works with "any provider." The `"dummy"` fallback means local
servers (which don't check the key) still work.

#### Block 2.2 — `chat()`

```python
def chat(self, messages, tools=None, tool_choice=None, response_format=None, **kwargs):
    params = {
        "model": self.model,
        "messages": messages,
        "temperature": self.temperature,
    }
    if self.max_tokens:
        params["max_tokens"] = self.max_tokens
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice or "auto"
    if response_format:
        params["response_format"] = response_format
    params.update(kwargs)
    return self._client.chat.completions.create(**params)
```

**Why it exists:** One LLM call. Builds the request dict and sends it.

- `max_tokens` only added if set (some servers reject it)
- `tools` + `tool_choice` only added if you have tools. `"auto"` = "call a
  tool if you want, or just answer"
- `**kwargs` passes anything else through (e.g. `top_p`)

Returns the *raw* response. The agent digs into `.choices[0].message`.

**Design point:** This is a *thin* wrapper. No retries, no parsing. The LLM
client is a swappable detail, not a core concept.

#### Block 2.3 — `chat_stream()`

```python
def chat_stream(self, messages, tools=None, **kwargs):
    params = {
        "model": self.model,
        "messages": messages,
        "temperature": self.temperature,
        "stream": True,
    }
    if tools:
        params["tools"] = tools
    if self.max_tokens:
        params["max_tokens"] = self.max_tokens
    params.update(kwargs)
    return self._client.chat.completions.create(**params)
```

**Why it exists:** Same as `chat()` but `stream=True` — yields delta chunks as
the model generates. Exposed for completeness; the agent loop doesn't use it
yet, but the capability is there.

---

### 6.3 `tool.py` — turning functions into tools

The most "magical" file. Goal: you write a normal Python function, the
framework figures out how to describe it to the LLM.

#### Block 3.1 — `_TYPE_MAP`

```python
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
```

**Why:** Python types and JSON Schema types use different words. `str` →
`"string"`, `int` → `"integer"`. This dict is the translation table.

#### Block 3.2 — `_python_type_to_json_schema()`

```python
def _python_type_to_json_schema(tp):
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        args = getattr(tp, "__args__", None)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}
    return {"type": "string"}
```

**Why:** Handles simple cases via the map, then generic types like
`list[str]`.

- `list[str].__origin__` is `list`, `list[str].__args__` is `(str,)`
- So `list[str]` → `{"type": "array", "items": {"type": "string"}}`
- Final `return {"type": "string"}` is a safe fallback for anything unrecognized

#### Block 3.3 — `_parse_docstring()`

```python
def _parse_docstring(docstring):
    if not docstring:
        return "", {}
    lines = docstring.strip().split("\n")
    description = lines[0].strip()
    params = {}
    in_args = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            m = re.match(r"(\w+)\s*(?:\(\w+\))?\s*:\s*(.+)", stripped)
            if m:
                params[m.group(1)] = m.group(2).strip()
            elif stripped and not stripped.startswith((" ", "\t")):
                in_args = False
    description = re.sub(r"\s*Args:.*$", "", description, flags=re.DOTALL).strip()
    return description, params
```

**Why:** The LLM needs to know what each parameter *means*, not just its type.
So we mine the docstring.

**Convention:**

- First line = the tool's description
- An `Args:` section where each line is `name (type): description`

The regex captures the param name (group 1), optionally skips `(type)`, grabs
the description (group 2). The final `re.sub` cleans the description in case
"Args:" leaked onto line 1.

**Design point:** It's a *convention*, not a requirement. Skip the docstring →
no descriptions, but the tool still works.

#### Block 3.4 — `_build_tool_schema()`

```python
def _build_tool_schema(func):
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties = {}
    required = []
    doc_description, param_docs = _parse_docstring(func.__doc__)
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        tp = hints.get(name, str)
        schema = _python_type_to_json_schema(tp)
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc_description or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
```

**Why:** The heart of it. Takes a function, produces the exact JSON the
OpenAI tool-calling API expects.

- `get_type_hints(func)` → the annotations (`{"query": str, "max_results": int}`)
- `inspect.signature(func)` → params in order, with defaults
- Per param: build schema, attach docstring description, mark `required` if no default
- Skip `self`/`cls`
- Output matches OpenAI: `{"type": "function", "function": {name, description, parameters}}`

#### Block 3.5 — The `Tool` class

```python
class Tool:
    def __init__(self, func, name=None, description=None):
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self.is_async = inspect.iscoroutinefunction(func)
        self.schema = _build_tool_schema(func)
        if name:
            self.schema["function"]["name"] = name
        if description:
            self.schema["function"]["description"] = description

    def to_openai_schema(self):
        return self.schema

    async def execute(self, **kwargs) -> str:
        if self.is_async:
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)
```

**Why:** A `Tool` is a *function plus metadata*. Wraps the raw function,
pre-computes the schema once.

- `name`/`description` overrides let you rename a tool without renaming the
  function (matters for `as_tool()` in multi-agent)
- `execute()` is how the LLM's tool call actually runs. Handles sync + async.
  **Always returns a string** — that's what goes back into the conversation.
  Dicts get JSON-serialized; `default=str` means non-serializable objects get
  `str()`-ed instead of crashing.

#### Block 3.6 — `__call__` + the `tool` decorator

```python
    def __call__(self, **kwargs) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.execute(**kwargs)).result()
        return asyncio.run(self.execute(**kwargs))


def tool(func=None, *, name=None, description=None):
    if func is not None:
        return Tool(func, name=name, description=description)
    def wrapper(f):
        return Tool(f, name=name, description=description)
    return wrapper
```

**Why `__call__`:** Makes a `Tool` directly callable with `tool(...)` for when
*you* (not the LLM) want to use it in a plain script.

**The async gotcha:** `asyncio.run()` can't be called from inside a running
event loop. So if we're already in an async context, we spin up a thread to
run it.

**Why the dual-mode decorator:** Supports both:

- `@tool` (bare) — `func` is the decorated function
- `@tool(name="x")` (with args) — `func` is `None`, so we return a `wrapper`
  that receives the function

Without the `func=None` default, the `@tool(name=...)` form wouldn't work.

---

### 6.4 `hooks.py` — the extension point

#### Block 4.1 — `HookFn` type

```python
HookFn = Callable[[str, HookEvent], HookEvent | dict[str, Any] | None]
```

**Why:** A hook is just a function with this signature. It gets the agent's
name + the current event, and can return one of three things:

- `None` — "nothing to say, keep going"
- a modified `HookEvent` — "here's a changed version"
- a `dict` of overrides — e.g. `{"abort": True, "message": "..."}`

#### Block 4.2 — `HookPipeline`

```python
class HookPipeline:
    def __init__(self, hooks=None):
        self.hooks = hooks or []
    def add(self, hook):
        self.hooks.append(hook)
    def run(self, agent_name, event):
        overrides = {}
        for hook in self.hooks:
            result = hook(agent_name, event)
            if result is None:
                continue
            if isinstance(result, HookEvent):
                event = result
            elif isinstance(result, dict):
                overrides.update(result)
        return event, overrides
```

**Why:** Runs *all* your hooks in order for a given event. Threads the
(possibly modified) event through each hook, accumulates override dicts.
Returns the final event + combined overrides. The agent checks overrides for
things like `abort`.

**Design point:** The whole "middleware" idea in ~20 lines. No plugin
registry, no config. A guardrail is a hook. A logger is a hook. A rate
limiter is a hook. Same shape → they compose trivially.

---

### 6.5 `memory.py` — remembering the conversation

#### Block 5.1 — `estimate_tokens()`

```python
def estimate_tokens(text):
    return max(1, len(text) // 4)
```

**Why:** We need to know roughly how big the conversation is to avoid blowing
the context window. A real tokenizer (`tiktoken`) would be a heavy dependency.
"4 chars ≈ 1 token" is good enough for budget management. `max(1, ...)` so
empty strings still count as something.

#### Block 5.2 — `message_tokens()`

```python
def message_tokens(msg):
    tokens = 4  # overhead per message
    content = msg.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                tokens += estimate_tokens(part.get("text", ""))
    for tc in msg.get("tool_calls", []):
        tokens += estimate_tokens(json.dumps(tc, default=str))
    if msg.get("tool_call_id"):
        tokens += estimate_tokens(msg["tool_call_id"])
    return tokens
```

**Why:** A message isn't always just a string. It can be:

- a plain string (`content`)
- a list of parts (multimodal — text + image)
- carry `tool_calls` (an assistant message that asked for tools)
- carry a `tool_call_id` (a tool result)

This counts all of them. The `+4` is per-message overhead the API charges
(role tags, separators).

#### Block 5.3 — `Memory` constructor + `add()`

```python
class Memory:
    def __init__(self, max_tokens=128_000, system_prompt=None):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.messages = []

    def add(self, msg):
        self.messages.append(msg)
        self._trim()
```

**Why:** Holds the conversation. `max_tokens` is the budget. `system_prompt`
is kept separately because it must *always* be first and never trimmed. Every
`add()` triggers a trim check — memory self-manages.

#### Block 5.4 — The convenience adders + `get_messages()`

```python
    def add_user(self, content):
        self.add({"role": "user", "content": content})
    def add_assistant(self, content=None, **kwargs):
        msg = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self.add(msg)
    def add_tool_result(self, tool_call_id, content):
        self.add({"role": "tool", "tool_call_id": tool_call_id, "content": content})
    def get_messages(self):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.messages)
        return msgs
```

**Why:** Build the right message shape. Note `add_assistant` takes `**kwargs`
— because when the LLM calls a tool, the assistant message needs a
`tool_calls` list attached, not just text. The `content=None` case is exactly
that: an assistant message that's *only* tool calls.

`get_messages()` returns the full list in API order: system prompt first, then
the conversation.

#### Block 5.5 — `total_tokens()` + `_trim()`

```python
    def total_tokens(self):
        total = 0
        if self.system_prompt:
            total += estimate_tokens(self.system_prompt)
        for msg in self.messages:
            total += message_tokens(msg)
        return total

    def _trim(self):
        while self.total_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.pop(0)
```

**Why:** When over budget, drop the *oldest* messages (from the front). The
`len > 2` guard means we never trim to nothing — we always keep the most
recent exchange.

**Design point:** This is the simplest possible strategy (FIFO). More
sophisticated frameworks do summarization or importance-based eviction, but
for a minimal framework, dropping old turns is enough.

---

### 6.6 `structured.py` — getting typed data back

#### Block 6.1 — `build_structured_prompt()`

```python
def build_structured_prompt(schema):
    schema_json = schema.model_json_schema()
    schema_json.pop("$schema", None)
    return (
        "Respond with ONLY a valid JSON object matching this schema. "
        "Do not include markdown code fences or any text outside the JSON.\n\n"
        f"Schema:\n{json.dumps(schema_json, indent=2)}"
    )
```

**Why:** Instead of relying on a provider-specific "structured outputs"
feature, we just *tell* the model the shape we want.
`schema.model_json_schema()` is Pydantic's built-in way to turn a model class
into a JSON Schema. We paste that into the prompt. Works with *any* LLM,
including local ones.

#### Block 6.2 — `extract_json()`

```python
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")
```

**Why:** LLMs are messy. They sometimes wrap JSON in ``` fences, or add
"Here's your answer:" before it. This tries, in order:

1. Parse the whole thing as JSON (best case)
2. Find JSON inside a code fence
3. Find the first `{...}` or `[...]` block anywhere

Defensive because real model output isn't always clean.

#### Block 6.3 — `parse_structured()`

```python
def parse_structured(text, schema):
    data = extract_json(text)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return schema.model_validate(data)
```

**Why:** Extract the JSON, unwrap a single-element list (models sometimes do
that), then validate against your Pydantic model. `model_validate` is what
actually enforces the types — if the model returned a string where you wanted
a float, this raises.

---

### 6.7 `agent.py` — the loop (the core)

Where it all comes together.

#### Block 7.1 — Constructor

```python
def __init__(self, name="agent", model="gpt-4o", system_prompt="...",
             tools=None, max_turns=15, temperature=0.7, max_tokens=None,
             context_window=128_000, api_key=None, base_url=None,
             hooks=None, on_token=None, **llm_kwargs):
    self.name = name
    self.max_turns = max_turns
    self.on_token = on_token
    self.llm = LLMClient(model=model, api_key=api_key, base_url=base_url,
                         temperature=temperature, max_tokens=max_tokens, **llm_kwargs)
    self.memory = Memory(max_tokens=context_window, system_prompt=system_prompt)
    self.tools = {}
    for t in (tools or []):
        self.tools[t.name] = t
    self.hook_pipeline = HookPipeline(hooks)
```

**Why:** The `Agent` wires together the three collaborators — an `LLMClient`,
a `Memory`, and a `HookPipeline`.

- `self.tools` is a *dict keyed by name* (not a list) because when the LLM
  says "call tool `calculate`", we look it up by name in O(1)
- `max_turns` is a safety valve so a misbehaving model can't loop forever

#### Block 7.2 — `_tool_schemas()` + `_emit()`

```python
def _tool_schemas(self):
    if not self.tools:
        return None
    return [t.to_openai_schema() for t in self.tools.values()]

def _emit(self, event_type, **kwargs):
    event = HookEvent(type=event_type, **kwargs)
    return self.hook_pipeline.run(self.name, event)
```

**Why:**

- `_tool_schemas()` converts our `Tool` objects into the list of JSON schemas
  the LLM needs. Returns `None` if no tools (so we don't send an empty
  `tools` param).
- `_emit()` is a tiny helper — build a `HookEvent` and push it through the
  pipeline. Every moment in the loop calls this. Returns `(event, overrides)`
  which the loop checks.

#### Block 7.3 — The sync/async split

```python
def run(self, user_input):
    import asyncio
    return asyncio.run(self.run_async(user_input))

async def run_async(self, user_input):
    ...
```

**Why:** The real logic is async (tools can be async). But you don't *want*
to write `async def` in your app code. So `run()` is a sync facade that calls
`asyncio.run()` on the async version. You get both: `agent.run(...)` for
simple use, `await agent.run_async(...)` if you're already in async code.

#### Block 7.4 — Loop setup

```python
_, overrides = self._emit(EventType.AGENT_START, data={"input": user_input})
if overrides.get("abort"):
    return AgentResult(text=overrides.get("message", "Aborted"), turns=0)
self.memory.add_user(user_input)
tool_schemas = self._tool_schemas()
all_tool_calls = []
total_usage = {}
final_text = ""
```

**Why:** Before the loop: fire `AGENT_START` (a guardrail can abort here —
that's how prompt-injection blocking works). Add the user's message to memory.
Pre-compute tool schemas. Set up accumulators.

#### Block 7.5 — One iteration: call the LLM

```python
for turn in range(1, self.max_turns + 1):
    messages = self.memory.get_messages()
    _, overrides = self._emit(EventType.LLM_CALL, turn=turn, messages=messages)
    if overrides.get("abort"):
        final_text = overrides.get("message", "Aborted")
        break
    try:
        response = self.llm.chat(messages, tools=tool_schemas)
    except Exception as e:
        _, _ = self._emit(EventType.ERROR, turn=turn, error=e)
        raise
    choice = response.choices[0]
    msg = choice.message
    if response.usage:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = getattr(response.usage, k, 0)
            if v:
                total_usage[k] = total_usage.get(k, 0) + v
    _, overrides = self._emit(EventType.LLM_RESPONSE, turn=turn, response=msg, messages=messages)
    if overrides.get("abort"):
        final_text = overrides.get("message", "Aborted")
        break
```

**Why:** One iteration. Grab current messages, fire `LLM_CALL` (another abort
point), call the LLM. If it throws, fire `ERROR` and re-raise. Pull out the
assistant's message, accumulate token usage across turns (each turn is a
separate API call, so we sum), fire `LLM_RESPONSE` (a hook can inspect/modify
the response).

#### Block 7.6 — The key branch: tool calls

```python
    if msg.tool_calls:
        self.memory.add_assistant(
            content=msg.content,
            tool_calls=[
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        )
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}
            tool_call_info = {"turn": turn, "tool": tool_name, "args": args}
            all_tool_calls.append(tool_call_info)
            _, overrides = self._emit(EventType.TOOL_CALL, turn=turn,
                                      tool_name=tool_name, tool_args=args)
            if overrides.get("abort"):
                final_text = overrides.get("message", "Aborted")
                break
```

**Why:** The model wants to call tools.

- We must record the assistant's message *including* the tool calls — required
  by the API's message format. The next messages (tool results) reference
  these `id`s. `content=msg.content` can be `None` (model sometimes emits
  tool calls with no text).
- The LLM sends tool arguments as a *JSON string*. We parse it into a dict.
  If parsing fails, we don't crash — we pass the raw string through so the
  tool can complain.
- Log the call, fire `TOOL_CALL` (a rate limiter could abort here).

#### Block 7.7 — Executing the tool + the exit

```python
            tool = self.tools.get(tool_name)
            if tool is None:
                result_str = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    result_str = await tool.execute(**args)
                except Exception as e:
                    result_str = f"Error executing {tool_name}: {e}"
            tool_call_info["result"] = result_str[:500]
            _, overrides = self._emit(EventType.TOOL_RESULT, turn=turn,
                                      tool_name=tool_name, tool_result=result_str)
            if overrides.get("abort"):
                final_text = overrides.get("message", "Aborted")
                break
            self.memory.add_tool_result(tc.id, result_str)
        if overrides.get("abort"):
            break
        continue

    final_text = msg.content or ""
    self.memory.add_assistant(content=final_text)
    break
```

**Why:**

- Look up the tool and run it. Two failure modes handled gracefully: unknown
  tool name, and the tool throwing. Both produce an *error string* that goes
  back to the LLM — so the model can react ("that tool failed, let me try
  differently") instead of the whole agent crashing.
- Fire `TOOL_RESULT`, append the result to memory as a `role: "tool"` message
  tied to the `tc.id`.
- After handling all tool calls this turn, `continue` — back to the top so
  the LLM sees the tool results.
- **The exit:** if there were *no* tool calls, the model gave its final
  answer. Save it, break.

#### Block 7.8 — Return + structured output

```python
    _, _ = self._emit(EventType.AGENT_END, data={"text": final_text, "turns": turn})
    return AgentResult(text=final_text, turns=turn, tool_calls=all_tool_calls, usage=total_usage)

async def run_structured_async(self, user_input, schema):
    original_prompt = self.memory.system_prompt
    self.memory.system_prompt = f"{original_prompt}\n\n{build_structured_prompt(schema)}"
    try:
        result = await self.run_async(user_input)
        result.structured = parse_structured(result.text, schema)
        return result
    finally:
        self.memory.system_prompt = original_prompt
```

**Why:**

- Fire `AGENT_END`, return the bundled result.
- `run_structured_async` temporarily appends the schema instructions to the
  system prompt, runs the normal loop, then parses the text into your Pydantic
  model. The `try/finally` guarantees the system prompt is restored even if
  parsing fails — so the agent stays reusable.

#### Block 7.9 — Multi-agent: `as_tool()`

```python
def as_tool(self, name=None, description=None):
    tool_name = name or f"agent_{self.name}"
    tool_desc = description or f"Delegate a task to the {self.name} agent."
    async def _agent_tool(task: str) -> str:
        saved_messages = self.memory.messages
        self.memory.messages = []
        try:
            result = await self.run_async(task)
            return result.text
        finally:
            self.memory.messages = saved_messages
    return Tool(_agent_tool, name=tool_name, description=tool_desc)
```

**Why:** Wraps the whole agent as a single tool. When the parent calls it, we:

1. Save the current conversation
2. Clear it (sub-agent starts fresh — no context pollution)
3. Run the sub-agent's own full loop on the `task`
4. Return just its text
5. Restore the original conversation

**Design point:** This is why multi-agent is "composition, not
orchestration." No special multi-agent machinery — an agent is just a tool
that happens to run its own loop. The parent doesn't know or care that it's
talking to another agent.

---

### 6.8 `__init__.py` — the public API

#### Block 8.1 — Re-exports

```python
from .agent import Agent
from .hooks import HookFn, HookPipeline
from .llm import LLMClient
from .memory import Memory
from .structured import build_structured_prompt, parse_structured
from .tool import Tool, tool
from .types import AgentResult, EventType, HookEvent

__all__ = [...]
```

**Why:** Makes `from agent_core import Agent, tool` work. Re-exports the
public names so users don't have to know the internal module layout.
`__all__` controls what `from agent_core import *` pulls in.

---

### 6.9 How it fits

#### Block 9.1 — The wiring diagram

```
Your code
   │
   │  @tool  ──────────────►  tool.py  (function → JSON schema)
   │
   │  Agent(tools=[...])
   │
   ▼
agent.py  ◄── the loop
   │
   ├──► llm.py        (talks to the model)
   ├──► memory.py     (remembers the conversation, trims it)
   ├──► hooks.py      (lets you intercept every moment)
   ├──► structured.py (optional: typed output)
   └──► types.py      (the shared vocabulary)
```

**The whole thing is ~1,200 lines.** The actual loop in `agent.py` is about
80 lines. Everything else is support structure.

**The mental model:**

```
You write:
  1. A system prompt   (what the agent is)
  2. Some @tool funcs  (what it can do)
  3. Optional hooks    (what it must/mustn't do)

The framework handles:
  - The loop (LLM → tool calls → results → LLM → ...)
  - Schema generation for tools
  - Context window management
  - Error handling
  - Structured output parsing
```

---

## 7. Development Approach

### Coding Agent Used: **Claude** (Anthropic's coding agent, via the pi CLI)

### How I Worked

The framework was built iteratively in ~4 hours, working with Claude as a pair
programmer:

1. **Scaffold (30 min)** — Defined the core abstractions: what is an Agent,
   what is a Tool, what is a Hook. Wrote the type system and module structure.
   Claude helped validate the API surface and catch design issues early.

2. **Tool system (45 min)** — Built the `@tool` decorator with schema
   introspection. The tricky part was docstring parsing for parameter
   descriptions. Claude caught the naming conflict between the `Tool` class
   and the `@tool` decorator, and the edge case where custom tool names need
   to override the schema's function name.

3. **Agent loop (45 min)** — Implemented the core observe→think→act cycle.
   Claude helped with the OpenAI tool-calling message format (assistant
   message with `tool_calls` array, followed by `role: "tool"` messages with
   `tool_call_id`). Tested against a local llama.cpp server running
   Qwen3.8-27B.

4. **Structured output (20 min)** — Added Pydantic schema injection into the
   system prompt and JSON extraction from LLM responses. Claude suggested
   handling markdown code fences and extra text around the JSON.

5. **Hooks (20 min)** — Added the hook pipeline. Claude helped design the
   event system (typed events with optional data) and the override mechanism
   (hooks can return `{"abort": True}` to stop the loop).

6. **Multi-agent (15 min)** — Added `agent.as_tool()` for composition. Claude
   caught the bug where the internal function name was leaking into the tool
   schema instead of the custom name.

7. **Examples & validation (30 min)** — Built the research agent and
   guardrails examples. Ran end-to-end tests against the local LLM server to
   verify tool calling, structured output, multi-agent delegation, and hook
   interception all work.

### Validation

- ✅ Tool calling: Agent correctly calls `calculate("15 * 23 + 7")` → gets `352`
- ✅ Structured output: `run_structured()` returns validated Pydantic model
- ✅ Multi-agent: Writer agent delegates to Researcher agent via `as_tool()`
- ✅ Hooks: Guardrail blocks prompt injection, rate limiter caps tool calls
- ✅ Memory: Token-aware truncation keeps context within budget
- ✅ Works with any OpenAI-compatible endpoint (tested with llama.cpp + Qwen3.8-27B)

---

## 8. Requirements & License

- Python 3.10+
- An OpenAI-compatible API (OpenAI, Ollama, vLLM, llama.cpp server, etc.)

**License:** MIT
