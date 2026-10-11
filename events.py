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
    """Payload handed to hooks at each event. Only relevant fields are set.

    Fields:
        type: which event this is (see EventType).
        turn: the loop turn number (1-based).
        messages: the conversation sent to / received from the LLM.
        response: the assistant message that came back from the LLM.
        tool_name / tool_args / tool_result: set on TOOL_CALL / TOOL_RESULT.
        error: set on ERROR events.
        data: catch-all for event-specific extras. Hooks read/write this —
            e.g. data["abort"] stops the run, data["handoff"] routes to
            another agent, data["final_text"] carries the final answer.
    """

    type: EventType
    turn: int = 0
    messages: list = None
    response: Any = None
    tool_name: str = None
    tool_args: dict = None
    tool_result: Any = None
    error: Exception = None
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Return value of agent.run() — the receipt for one agent run.

    Fields:
        text: the final answer text (or the handoff marker).
        turns: how many LLM turns the run took.
        tool_calls: list of {name, args} dicts for every tool call made.
        structured: optional structured output (if the agent produced one).
        usage: token usage (prompt/completion/total).
        aborted: True if the run stopped early (e.g. hit the tool-call limit
            or a hook set abort) instead of producing a normal final answer.
        abort_reason: why it stopped, when aborted.
        handoff: the name of the agent to hand off to (multi-agent), or None.
    """

    text: str
    turns: int
    tool_calls: list = field(default_factory=list)
    structured: Any = None
    usage: dict = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = None
    handoff: str = None
