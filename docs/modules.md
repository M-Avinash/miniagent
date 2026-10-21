# Modules

| Module | Purpose |
|--------|---------|
| `agent.py` | Agentic loop (reason → act → observe), bounded by `max_tool_calls` |
| `llm.py` | Provider-agnostic LLM client, retry on 429 |
| `tool.py` | `@tool` decorator — type hints + docstring → JSON schema; `make_handoff_tool` |
| `memory.py` | Bounded conversation window (sliding, system prompt pinned) |
| `hooks.py` | Event hooks — inspect, mutate, or abort at any loop step; `handoff_hook` |
| `events.py` | Event types and schema definitions |
| `structured.py` | JSON extraction + field validation |
| `sandbox.py` | `run_python` — isolated subprocess (minimal env, temp cwd, timeout) |
| `observe.py` | `verbose_log` hook — labeled event spans per agent |
| `observation.py` | ObservationalMemory — distilled cross-run JSONL store |
| `orchestrator.py` | HandoffOrchestrator, handoff/abort markers, session context |
| `graph_memory.py` | GraphMemory — entity key tracking for relational schemas |
