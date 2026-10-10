# agent-core — Block-by-Block Code Walkthrough

> Interactive walkthrough. We go one block at a time. You say **"next"** and we move forward.
>
> **How it works:** Each section = one code block. I show the code + explain *why* it exists and how it fits. You say "next" to advance. Say "why?" or "deeper" if you want me to expand on a block. Say "skip" to jump ahead.

---

## The Roadmap

We walk the framework in **dependency order** — each file only depends on files before it.

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

## 0. The Big Picture

### Block 0.1 — What an agent actually is

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

> **Say "next" to start with `types.py`.**

---

## 1. `types.py` — the shared vocabulary

This is the "common language" the rest of the code shares. No logic here — just data shapes.

### Block 1.1 — `EventType`

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

**Why it exists:** The agent loop has several distinct moments (starting, calling the LLM, getting a response, calling a tool, getting a tool result, finishing). Hooks need to know *which moment* they're reacting to. This enum names those moments.

**Why `str` enum:** `EventType.TOOL_CALL == "tool_call"` is `True`. Easy to compare, print, and log.

> **next →**

### Block 1.2 — `HookEvent`

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

**Why it exists:** This is the *payload* handed to hooks. One object that carries everything a hook might want about the current moment.

- `type` — which moment (required, no default)
- `turn` — which loop iteration
- `messages` / `response` — set on LLM events
- `tool_name` / `tool_args` / `tool_result` — set on tool events
- `error` — set on error events
- `data` — catch-all dict (e.g. the user's input on `AGENT_START`)

Most fields are `None` because a given event only fills in the relevant ones. `@dataclass` auto-generates the constructor — no boilerplate.

> **next →**

### Block 1.3 — `AgentResult`

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

**Gotcha:** `field(default_factory=list)` instead of `= []`. A mutable default like `= []` would make every instance share the *same* list. `default_factory` gives each instance its own fresh list.

> **next →** (moves to `llm.py`)

---

## 2. `llm.py` — talking to the model

The only file that touches the outside world. Everything else is pure Python.

### Block 2.1 — Constructor

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

**Why it exists:** Wraps the `openai` package so the rest of the framework never knows about OpenAI's specifics.

**The key trick — `base_url`:** The OpenAI client is *protocol-based*, not company-based. Point it at a different `base_url` and it talks to whatever speaks the OpenAI format — Ollama, vLLM, llama.cpp, Groq, Together. That's how one client works with "any provider." The `"dummy"` fallback means local servers (which don't check the key) still work.

> **next →**

### Block 2.2 — `chat()`

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
- `tools` + `tool_choice` only added if you have tools. `"auto"` = "call a tool if you want, or just answer"
- `**kwargs` passes anything else through (e.g. `top_p`)

Returns the *raw* response. The agent digs into `.choices[0].message`.

**Design point:** This is a *thin* wrapper. No retries, no parsing. The LLM client is a swappable detail, not a core concept.

> **next →**

### Block 2.3 — `chat_stream()`

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

**Why it exists:** Same as `chat()` but `stream=True` — yields delta chunks as the model generates. Exposed for completeness; the agent loop doesn't use it yet, but the capability is there.

> **next →** (moves to `tool.py`)

---

## 3. `tool.py` — turning functions into tools

The most "magical" file. Goal: you write a normal Python function, the framework figures out how to describe it to the LLM.

### Block 3.1 — `_TYPE_MAP`

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

**Why:** Python types and JSON Schema types use different words. `str` → `"string"`, `int` → `"integer"`. This dict is the translation table.

> **next →**

### Block 3.2 — `_python_type_to_json_schema()`

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

**Why:** Handles simple cases via the map, then generic types like `list[str]`.

- `list[str].__origin__` is `list`, `list[str].__args__` is `(str,)`
- So `list[str]` → `{"type": "array", "items": {"type": "string"}}`
- Final `return {"type": "string"}` is a safe fallback for anything unrecognized

> **next →**

### Block 3.3 — `_parse_docstring()`

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

**Why:** The LLM needs to know what each parameter *means*, not just its type. So we mine the docstring.

**Convention:**
- First line = the tool's description
- An `Args:` section where each line is `name (type): description`

The regex captures the param name (group 1), optionally skips `(type)`, grabs the description (group 2). The final `re.sub` cleans the description in case "Args:" leaked onto line 1.

**Design point:** It's a *convention*, not a requirement. Skip the docstring → no descriptions, but the tool still works.

> **next →**

### Block 3.4 — `_build_tool_schema()`

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

**Why:** The heart of it. Takes a function, produces the exact JSON the OpenAI tool-calling API expects.

- `get_type_hints(func)` → the annotations (`{"query": str, "max_results": int}`)
- `inspect.signature(func)` → params in order, with defaults
- Per param: build schema, attach docstring description, mark `required` if no default
- Skip `self`/`cls`
- Output matches OpenAI: `{"type": "function", "function": {name, description, parameters}}`

> **next →**

### Block 3.5 — The `Tool` class

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

**Why:** A `Tool` is a *function plus metadata*. Wraps the raw function, pre-computes the schema once.

- `name`/`description` overrides let you rename a tool without renaming the function (matters for `as_tool()` in multi-agent)
- `execute()` is how the LLM's tool call actually runs. Handles sync + async. **Always returns a string** — that's what goes back into the conversation. Dicts get JSON-serialized; `default=str` means non-serializable objects get `str()`-ed instead of crashing.

> **next →**

### Block 3.6 — `__call__` + the `tool` decorator

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

**Why `__call__`:** Makes a `Tool` directly callable with `tool(...)` for when *you* (not the LLM) want to use it in a plain script.

**The async gotcha:** `asyncio.run()` can't be called from inside a running event loop. So if we're already in an async context, we spin up a thread to run it.

**Why the dual-mode decorator:** Supports both:
- `@tool` (bare) — `func` is the decorated function
- `@tool(name="x")` (with args) — `func` is `None`, so we return a `wrapper` that receives the function

Without the `func=None` default, the `@tool(name=...)` form wouldn't work.

> **next →** (moves to `hooks.py`)

---

## 4. `hooks.py` — the extension point

### Block 4.1 — `HookFn` type

```python
HookFn = Callable[[str, HookEvent], HookEvent | dict[str, Any] | None]
```

**Why:** A hook is just a function with this signature. It gets the agent's name + the current event, and can return one of three things:
- `None` — "nothing to say, keep going"
- a modified `HookEvent` — "here's a changed version"
- a `dict` of overrides — e.g. `{"abort": True, "message": "..."}`

> **next →**

### Block 4.2 — `HookPipeline`

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

**Why:** Runs *all* your hooks in order for a given event. Threads the (possibly modified) event through each hook, accumulates override dicts. Returns the final event + combined overrides. The agent checks overrides for things like `abort`.

**Design point:** The whole "middleware" idea in ~20 lines. No plugin registry, no config. A guardrail is a hook. A logger is a hook. A rate limiter is a hook. Same shape → they compose trivially.

> **next →** (moves to `memory.py`)

---

## 5. `memory.py` — remembering the conversation

### Block 5.1 — `estimate_tokens()`

```python
def estimate_tokens(text):
    return max(1, len(text) // 4)
```

**Why:** We need to know roughly how big the conversation is to avoid blowing the context window. A real tokenizer (`tiktoken`) would be a heavy dependency. "4 chars ≈ 1 token" is good enough for budget management. `max(1, ...)` so empty strings still count as something.

> **next →**

### Block 5.2 — `message_tokens()`

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

This counts all of them. The `+4` is per-message overhead the API charges (role tags, separators).

> **next →**

### Block 5.3 — `Memory` constructor + `add()`

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

**Why:** Holds the conversation. `max_tokens` is the budget. `system_prompt` is kept separately because it must *always* be first and never trimmed. Every `add()` triggers a trim check — memory self-manages.

> **next →**

### Block 5.4 — The convenience adders + `get_messages()`

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

**Why:** Build the right message shape. Note `add_assistant` takes `**kwargs` — because when the LLM calls a tool, the assistant message needs a `tool_calls` list attached, not just text. The `content=None` case is exactly that: an assistant message that's *only* tool calls.

`get_messages()` returns the full list in API order: system prompt first, then the conversation.

> **next →**

### Block 5.5 — `total_tokens()` + `_trim()`

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

**Why:** When over budget, drop the *oldest* messages (from the front). The `len > 2` guard means we never trim to nothing — we always keep the most recent exchange.

**Design point:** This is the simplest possible strategy (FIFO). More sophisticated frameworks do summarization or importance-based eviction, but for a minimal framework, dropping old turns is enough.

> **next →** (moves to `structured.py`)

---

## 6. `structured.py` — getting typed data back

### Block 6.1 — `build_structured_prompt()`

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

**Why:** Instead of relying on a provider-specific "structured outputs" feature, we just *tell* the model the shape we want. `schema.model_json_schema()` is Pydantic's built-in way to turn a model class into a JSON Schema. We paste that into the prompt. Works with *any* LLM, including local ones.

> **next →**

### Block 6.2 — `extract_json()`

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

**Why:** LLMs are messy. They sometimes wrap JSON in ``` fences, or add "Here's your answer:" before it. This tries, in order:
1. Parse the whole thing as JSON (best case)
2. Find JSON inside a code fence
3. Find the first `{...}` or `[...]` block anywhere

Defensive because real model output isn't always clean.

> **next →**

### Block 6.3 — `parse_structured()`

```python
def parse_structured(text, schema):
    data = extract_json(text)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return schema.model_validate(data)
```

**Why:** Extract the JSON, unwrap a single-element list (models sometimes do that), then validate against your Pydantic model. `model_validate` is what actually enforces the types — if the model returned a string where you wanted a float, this raises.

> **next →** (moves to `agent.py` — the core)

---

## 7. `agent.py` — the loop (the core)

Where it all comes together.

### Block 7.1 — Constructor

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

**Why:** The `Agent` wires together the three collaborators — an `LLMClient`, a `Memory`, and a `HookPipeline`.

- `self.tools` is a *dict keyed by name* (not a list) because when the LLM says "call tool `calculate`", we look it up by name in O(1)
- `max_turns` is a safety valve so a misbehaving model can't loop forever

> **next →**

### Block 7.2 — `_tool_schemas()` + `_emit()`

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
- `_tool_schemas()` converts our `Tool` objects into the list of JSON schemas the LLM needs. Returns `None` if no tools (so we don't send an empty `tools` param).
- `_emit()` is a tiny helper — build a `HookEvent` and push it through the pipeline. Every moment in the loop calls this. Returns `(event, overrides)` which the loop checks.

> **next →**

### Block 7.3 — The sync/async split

```python
def run(self, user_input):
    import asyncio
    return asyncio.run(self.run_async(user_input))

async def run_async(self, user_input):
    ...
```

**Why:** The real logic is async (tools can be async). But you don't *want* to write `async def` in your app code. So `run()` is a sync facade that calls `asyncio.run()` on the async version. You get both: `agent.run(...)` for simple use, `await agent.run_async(...)` if you're already in async code.

> **next →**

### Block 7.4 — Loop setup

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

**Why:** Before the loop: fire `AGENT_START` (a guardrail can abort here — that's how prompt-injection blocking works). Add the user's message to memory. Pre-compute tool schemas. Set up accumulators.

> **next →**

### Block 7.5 — One iteration: call the LLM

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

**Why:** One iteration. Grab current messages, fire `LLM_CALL` (another abort point), call the LLM. If it throws, fire `ERROR` and re-raise. Pull out the assistant's message, accumulate token usage across turns (each turn is a separate API call, so we sum), fire `LLM_RESPONSE` (a hook can inspect/modify the response).

> **next →**

### Block 7.6 — The key branch: tool calls

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

- We must record the assistant's message *including* the tool calls — required by the API's message format. The next messages (tool results) reference these `id`s. `content=msg.content` can be `None` (model sometimes emits tool calls with no text).
- The LLM sends tool arguments as a *JSON string*. We parse it into a dict. If parsing fails, we don't crash — we pass the raw string through so the tool can complain.
- Log the call, fire `TOOL_CALL` (a rate limiter could abort here).

> **next →**

### Block 7.7 — Executing the tool + the exit

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

- Look up the tool and run it. Two failure modes handled gracefully: unknown tool name, and the tool throwing. Both produce an *error string* that goes back to the LLM — so the model can react ("that tool failed, let me try differently") instead of the whole agent crashing.
- Fire `TOOL_RESULT`, append the result to memory as a `role: "tool"` message tied to the `tc.id`.
- After handling all tool calls this turn, `continue` — back to the top so the LLM sees the tool results.
- **The exit:** if there were *no* tool calls, the model gave its final answer. Save it, break.

> **next →**

### Block 7.8 — Return + structured output

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
- `run_structured_async` temporarily appends the schema instructions to the system prompt, runs the normal loop, then parses the text into your Pydantic model. The `try/finally` guarantees the system prompt is restored even if parsing fails — so the agent stays reusable.

> **next →**

### Block 7.9 — Multi-agent: `as_tool()`

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

**Design point:** This is why multi-agent is "composition, not orchestration." No special multi-agent machinery — an agent is just a tool that happens to run its own loop. The parent doesn't know or care that it's talking to another agent.

> **next →** (moves to `__init__.py`)

---

## 8. `__init__.py` — the public API

### Block 8.1 — Re-exports

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

**Why:** Makes `from agent_core import Agent, tool` work. Re-exports the public names so users don't have to know the internal module layout. `__all__` controls what `from agent_core import *` pulls in.

> **next →** (final block)

---

## 9. How it fits

### Block 9.1 — The wiring diagram

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

**The whole thing is ~1,200 lines.** The actual loop in `agent.py` is about 80 lines. Everything else is support structure.

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

## Controls (cheat sheet)

| You say | I do |
|---------|------|
| `next` | Show the next block |
| `why?` | Expand on the current block's reasoning |
| `deeper` | Go into the implementation details / edge cases |
| `example` | Show a concrete usage example of the current block |
| `skip` | Jump to the next file |
| `back` | Re-show the previous block |
| `where am i` | Show current position in the roadmap |
