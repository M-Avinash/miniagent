# miniagent

A minimal agent framework. Built with pi coding agent and a self-hosted Qwen3.8 27B LLM (llama.cpp, RTX 3090). The agentic loop lives in `agent.py`; everything else is a tool, a hook, or a wrapper. Runs on any OpenAI-compatible API. Scales from single agent to multi-agent handoff chains.

## Modules

| Module | Purpose |
|--------|---------|
| `agent.py` | Agentic loop (reason → act → observe), bounded by `max_tool_calls` |
| `llm.py` | Provider-agnostic LLM client, retry on 429 |
| `tool.py` | `@tool` decorator — type hints + docstring → JSON schema |
| `memory.py` | Bounded conversation window (sliding, system prompt pinned) |
| `hooks.py` | Event hooks — inspect, mutate, or abort at any loop step |
| `events.py` | Event types and schema definitions |
| `structured.py` | JSON extraction + field validation |
| `sandbox.py` | `run_python` — subprocess execution with timeout |
| `observe.py` | `verbose_log` hook — labeled event spans per agent |
| `observation.py` | ObservationalMemory — distilled cross-run JSONL store |
| `orchestrator.py` | HandoffOrchestrator, make_handoff_tool, handoff_hook |
| `graph_memory.py` | GraphMemory — entity key tracking for relational schemas |

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

## Setup & Run

```bash
git clone <repo-url>
cd miniagent
pip install -r requirements.txt

# Configure (or export env vars)
cat > config.env <<EOF
OPENAI_API_KEY=sk-...
MINIAGENT_MODEL=gpt-5.6-luna
MINIAGENT_TEMPERATURE=0.7
MINIAGENT_TOOL_CHOICE=auto
MINIAGENT_REASONING_EFFORT=none
EOF

# Run the multi-agent Northwind QnA example
python -m miniagent.examples.northwind_qna
```

The example is a terminal chatbot over SAP Northwind OData — a three-agent handoff chain (Orchestrator → Data Fetcher → Data Analyst). Multi-turn: `shared_state` and `GraphMemory` persist across questions. Type `reset` to clear, `quit` to exit.

See [`Northwind_test_questions.md`](examples/Northwind_test_questions.md) for test questions.

## Project Structure

```
miniagent/
├── __init__.py
├── agent.py
├── llm.py
├── tool.py
├── memory.py
├── hooks.py
├── structured.py
├── events.py
├── sandbox.py
├── observe.py
├── observation.py
├── orchestrator.py
├── graph_memory.py
├── examples/
│   ├── northwind_qna.py
│   └── Northwind_test_questions.md
├── config.env          # gitignored
└── .env.example
```

## Requirements

- Python 3.10+
- See `requirements.txt`

## Configuration

Resolution order: constructor arg > env / `config.env` > default.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `dummy` | API key |
| `OPENAI_BASE_URL` | — | Base URL (set for local servers) |
| `MINIAGENT_MODEL` | `gpt-5.6-luna` | Model name |
| `MINIAGENT_TEMPERATURE` | `0.7` | Sampling temperature |
| `MINIAGENT_MAX_TOKENS` | — | Max tokens per call (needed for local models) |
| `MINIAGENT_TOOL_CHOICE` | `auto` | `auto` or `required` |
| `MINIAGENT_REASONING_EFFORT` | — | Set `none` for reasoning models |
| `MINIAGENT_MAX_RETRIES` | `3` | Retries on 429 |
