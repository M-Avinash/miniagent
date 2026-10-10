# miniagent — Build Plan

_A minimal, observable, proactive agent framework in Python. Built from scratch._

---

## 1. Philosophy

> The loop *is* the agent. An agent is: user input → LLM → (final answer | tool call) → repeat.

No graphs, no DAGs, no orchestration engine. Everything else is support structure
around that loop — and every capability is **opt-in**, slotted in as a tool,
hook, or wrapper. The core stays minimal.

**Out of scope (deliberately):**
- Graph/orchestration engine — deterministic pipelines are plain `if/else` orchestrator code
- Streaming in the agent loop — the LLM client supports it; a UI layer can add `on_token` later
- Raw conversation persistence — replaced by observational memory (#11)
- RAG / vector stores — tools can wrap any retrieval system
- MCP / A2A — not in scope; would slot in as opt-in modules later

---

## 2. Feature List

| # | Feature | Design |
|---|---------|--------|
| 1 | **Agent loop** | input → LLM → (answer \| tool call) → repeat; `max_turns` safety valve; sync `run()` + async `run_async()` |
| 2 | **`@tool` decorator** | Python function + type hints + docstring → auto JSON schema; sync & async funcs; rename/override via `@tool(name=..., description=...)` |
| 3 | **LLM client** | OpenAI-compatible (`OPENAI_BASE_URL` → OpenAI, Ollama, vLLM, llama.cpp); single `chat()` call. (Streaming: add `chat_stream()` later if a UI needs token-by-token output.) |
| 4 | **Memory** | in-process conversation list; ~4 chars/token estimation; FIFO trimming to a token budget; system prompt always preserved, never trimmed |
| 5 | **Structured output** | Pydantic schema injected into system prompt → JSON extracted (fence/text tolerant) → `model_validate`; agent restored via `try/finally` |
| 6 | **Hooks** | `HookPipeline` middleware; 7 event types (`AGENT_START`, `LLM_CALL`, `LLM_RESPONSE`, `TOOL_CALL`, `TOOL_RESULT`, `AGENT_END`, `ERROR`); hooks can observe, modify the event, or return overrides (`{"abort": True, ...}`) |
| 7 | **Multi-agent** | `agent.as_tool()` — sub-agent runs a fresh full loop, returns text; parent sees it as an ordinary tool |
| 8 | **Heartbeat** | `HeartbeatAgent` wrapper — runs a task on an interval (`run_forever`); optional maker-checker (second agent validates before "done") |
| 9 | **Sandbox** | `run_python` tool — executes code in a subprocess with timeout; stdout+stderr captured and returned; errors come back as strings, not crashes |
| 10 | **Observability** | `verbose_log` hook — prints every event (turn, tool, args, result size, token usage) with per-agent labeling; optional debug level (full payloads) and JSONL-to-file mode |
| 11 | **Observational memory** | brief persistence — `AGENT_END` hook distills recent turns into 1–3 line observations (≤~50 words); appended to a capped JSONL store (FIFO, e.g. last 50); last N injected into system prompt on next run; no embeddings, no RAG |

### Public API (target)

```python
from miniagent import (
    Agent, Tool, tool,
    LLMClient, Memory,
    HookEvent, HookFn, HookPipeline, EventType, AgentResult,
    build_structured_prompt, parse_structured,
    HeartbeatAgent,
    run_python,          # sandbox tool
    verbose_log,         # observability hook
    ObservationalMemory, # brief persistence
)
```

---

## 3. File Layout

```
miniagent/
├── __init__.py          # public API exports
├── events.py            # EventType, HookEvent, AgentResult
├── llm.py               # LLMClient (OpenAI-compatible wrapper)
├── .env.example         # committed config template
├── config.env           # local overrides (gitignored)
└── .gitignore
├── tool.py              # @tool decorator, Tool class, schema introspection
├── hooks.py             # HookPipeline (middleware)
├── memory.py            # Memory (context management + token estimation)
├── structured.py        # structured output via Pydantic
├── agent.py             # Agent (the loop + as_tool for multi-agent)
├── heartbeat.py         # HeartbeatAgent (feature 8)
├── sandbox.py           # run_python tool (feature 9)
├── observe.py           # verbose_log hook (feature 10)
└── observation.py       # ObservationalMemory (feature 11)
```

---

## 4. Build Order (dependency order)

| Block | File | Contents | Feature |
|-------|------|----------|---------|
| 1 | `events.py` | `EventType` enum, `HookEvent` dataclass, `AgentResult` dataclass | 6 (vocabulary) |
| 2 | `llm.py` | `LLMClient.__init__`, `chat()`, `chat_stream()` | 3 |
| 3 | `tool.py` | `_TYPE_MAP`, `_python_type_to_json_schema()`, `_parse_docstring()`, `_build_tool_schema()`, `Tool` class, `Tool.__call__`, `@tool` decorator | 2 |
| 4 | `hooks.py` | `HookFn` type, `HookPipeline` (runs hooks in sequence, threads event, accumulates overrides) | 6 |
| 5 | `memory.py` | `Memory` (bounded list, default max_messages=50 to hold a full 10-round run, system prompt pinned; no token counting) | 4 |
| 6 | `structured.py` | `build_structured_prompt()`, `extract_json()`, `parse_structured()` | 5 |
| 7 | `agent.py` | `Agent` (constructor, `run`/`run_async` loop, `run_structured`, `as_tool`) — **the ONE tool-calling loop (unified, tools always available, `auto` after turn 1, repetition guard). Single source of truth; playground Section 3 shrinks to call `agent.run()`.** | 1, 7 |
| 8 | `heartbeat.py` | `HeartbeatAgent` wrapper (`run_forever`, optional checker agent) | 8 |
| 9 | `sandbox.py` | `run_python` tool (subprocess + timeout, errors as strings) | 9 |
| 10 | `observe.py` | `verbose_log` hook (per-agent labeled event printing, debug level, JSONL-to-file mode) | 10 |
| 11 | `observation.py` | `ObservationalMemory` (AGENT_END distillation hook + capped JSONL store + prompt injection) | 11 |
| 12 | `__init__.py` | Re-export the public API | — |

### Capability design notes

**8 — Heartbeat**

```python
class HeartbeatAgent:
    def __init__(self, agent: Agent, interval_seconds: int, task: str,
                 checker: Agent | None = None):
        ...
    async def run_forever(self):
        while True:
            result = await self.agent.run_async(self.task)
            if self.checker is not None:          # maker-checker
                verdict = await self.checker.run_structured(
                    f"Validate this output is acceptable:\n{result.text}",
                    schema=CheckResult)
                # act on verdict (notify / retry / log)
            await asyncio.sleep(self.interval)
```

**9 — Sandbox**

```python
@tool
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess.
    Args:
        code (str): The Python code to execute.
    """
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=30)
    return (result.stdout or "") + (result.stderr or "")
```

Timeout/exception → return an error string, never raise (the LLM can react).

**10 — Observability**

```python
def verbose_log(agent_name: str, event: HookEvent, *,
                level: str = "info", log_file: str | None = None) -> None:
    # one line per event: turn, tool, args, result size, token usage
    # level="debug" → full payloads
    # log_file → also append JSON lines to file
```

**11 — Observational memory (brief)**

```python
class ObservationalMemory:
    def __init__(self, path: str = "observations.jsonl",
                 max_observations: int = 50, inject_last: int = 10):
        ...
    def hook(self, agent_name: str, event: HookEvent):
        if event.type == EventType.AGENT_END:
            obs = self._distill(event)            # 1–3 lines, ≤ ~50 words
            self.append(obs)                       # capped FIFO JSONL
    def prompt_section(self) -> str:
        return "Past observations:\n- ..."         # injected into system prompt
```

Distillation: one cheap LLM call (or rule-based fallback) summarizing the last
few turns. No embeddings, no RAG — just "inject last N."

---

## 5. Examples (later — an interesting one)

To be designed. Candidate: an **autonomous agent** combining heartbeat +
sandbox + verbose observability + observational memory — e.g. a periodic
"improve this codebase" agent that writes code, runs it in the sandbox,
observes the results, and remembers what it learned across runs.

---

## 6. Packaging (last)

- `README.md` — architecture, capabilities, quick start, "why no graph engine"
- `pyproject.toml` — dependencies: `openai`, `pydantic`
- `.gitignore`

---

## 7. Status

| Block | Status |
|-------|--------|
| 1. `events.py` | ✅ |
| 2. `llm.py` | ✅ |
| 3. `tool.py` | ✅ |
| 4. `hooks.py` | ✅ |
| 5. `memory.py` | ✅ |
| 6. `structured.py` | ✅ |
| 7. `agent.py` | ✅ |
| 8. `heartbeat.py` | ⬜ |
| 9. `sandbox.py` | ⬜ |
| 10. `observe.py` | ⬜ |
| 11. `observation.py` | ⬜ |
| 12. `__init__.py` | ⬜ |
