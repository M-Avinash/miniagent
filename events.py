"""Shared types for the agent framework. No logic here — just data shapes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Named moments in the agent loop that hooks can react to."""

    AGENT_START = "agent_start"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY = "memory"
    AGENT_END = "agent_end"
    ERROR = "error"


@dataclass
class HookEvent:
    """Payload handed to hooks at each event. Only relevant fields are set."""

    type: EventType
    turn: int = 0
    # LLM-related
    messages: list = None       # conversation sent to / received from the LLM
    response: Any = None        # the assistant message that came back
    # Tool-related
    tool_name: str = None
    tool_args: dict = None
    tool_result: Any = None
    # Error-related
    error: Exception = None
    # Catch-all for event-specific extras
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Return value of agent.run() — the receipt for one agent run."""

    text: str
    turns: int
    tool_calls: list = field(default_factory=list)
    structured: Any = None
    usage: dict = field(default_factory=dict)
    # Set when the run stopped early (e.g. hit the tool-call limit) instead of
    # producing a normal final answer. A normal run leaves these at their defaults.
    aborted: bool = False
    abort_reason: str = None
