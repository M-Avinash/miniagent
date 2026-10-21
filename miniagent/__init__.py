"""miniagent — a minimal agent framework.

The agentic loop lives in agent.py; everything else is a tool, a hook,
or a wrapper.

Public API:
    Agent, LLMClient, Memory, Tool, tool
    HookEvent, HookFn, HookPipeline, EventType, AgentResult
    build_structured_prompt, extract_json, parse_structured
    run_python
    verbose_log
    ObservationalMemory
"""

from .events import AgentResult, EventType, HookEvent
from .hooks import HookFn, HookPipeline, handoff_hook
from .llm import LLMClient
from .memory import Memory
from .tool import Tool, tool, make_handoff_tool
from .agent import Agent, make_supervisor_tool, make_agent_tool
from .structured import build_structured_prompt, extract_json, parse_structured
from .sandbox import run_python
from .observe import verbose_log, trace_hook
from .observation import ObservationalMemory
from .orchestrator import (
    HandoffOrchestrator,
    make_handoff_marker, parse_handoff,
    make_abort_marker, parse_abort,
)
from .graph_memory import GraphMemory, graph_memory_hook, graph_injection_hook

__all__ = [
    "Agent", "LLMClient", "Memory", "Tool", "tool",
    "make_supervisor_tool", "make_agent_tool",
    "HookEvent", "HookFn", "HookPipeline", "EventType", "AgentResult",
    "build_structured_prompt", "extract_json", "parse_structured",
    "run_python",
    "verbose_log", "trace_hook",
    "ObservationalMemory",
    "HandoffOrchestrator",
    "make_handoff_tool",
    "handoff_hook",
    "GraphMemory",
    "graph_memory_hook",
    "graph_injection_hook",
    "make_handoff_marker", "parse_handoff",
    "make_abort_marker", "parse_abort",
]
