"""miniagent — a lean, robust agent framework.

The loop is the orchestration. One tool-calling loop in agent.py, everything
else is a tool, a hook, or a wrapper.

Public API:
    Agent, LLMClient, Memory, Tool, tool
    HookEvent, HookFn, HookPipeline, EventType, AgentResult
    build_structured_prompt, extract_json, parse_structured
    HeartbeatAgent
    run_python
    verbose_log
    ObservationalMemory
"""

from .events import AgentResult, EventType, HookEvent
from .hooks import HookFn, HookPipeline
from .llm import LLMClient
from .memory import Memory
from .tool import Tool, tool
from .agent import Agent
from .structured import build_structured_prompt, extract_json, parse_structured
from .heartbeat import HeartbeatAgent
from .sandbox import run_python
from .observe import verbose_log
from .observation import ObservationalMemory
from .orchestrator import HandoffOrchestrator

__all__ = [
    # Core
    "Agent", "LLMClient", "Memory", "Tool", "tool",
    # Events & hooks
    "HookEvent", "HookFn", "HookPipeline", "EventType", "AgentResult",
    # Structured output
    "build_structured_prompt", "extract_json", "parse_structured",
    # Capabilities
    "HeartbeatAgent",
    "run_python",
    "verbose_log",
    "ObservationalMemory",
    "HandoffOrchestrator",
]
