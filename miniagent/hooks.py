"""HookPipeline — a tiny middleware engine for the agent loop.

A hook is a function that receives a HookEvent, may inspect or mutate it,
and optionally returns a replacement event. Hooks run in order. A buggy
hook is caught and skipped, so it can never crash the agent.

Convention: a hook stops the agent by setting  event.data["abort"] = True
(optionally with event.data["abort_reason"]). The agent loop checks that
flag after each emit().
"""

import sys
from typing import Callable, Optional

from .events import EventType, HookEvent

HookFn = Callable[[HookEvent], Optional[HookEvent]]


class HookPipeline:
    """Runs an ordered list of hooks over a HookEvent."""

    def __init__(self, hooks: list = None):
        self.hooks: list = list(hooks) if hooks else []

    def add(self, hook: HookFn) -> "HookPipeline":
        """Add a hook. Returns self so calls can be chained."""
        self.hooks.append(hook)
        return self

    def emit(self, event: HookEvent) -> HookEvent:
        """Run every hook over the event, in order, and return the result.

        A hook may return a replacement event (which becomes the new event
        for the remaining hooks) or None (keep the current one). A hook that
        raises is caught, logged to stderr, and skipped — it never crashes
        the agent.
        """
        for hook in self.hooks:
            try:
                result = hook(event)
            except Exception as ex:
                name = getattr(hook, "__name__", repr(hook))
                print(f"[hooks] {name} crashed: {ex!r} — skipped", file=sys.stderr)
                continue
            if result is not None:
                event = result
        return event


def handoff_hook(event):
    """Hook that reads a handoff tool call and sets event.data['handoff'].

    Attach this to an agent's hooks. When the agent calls the handoff tool,
    this hook sets event.data['handoff'] to the target agent name and aborts
    the agent's run (so the orchestrator can route to the next agent).
    """
    if event.type == EventType.TOOL_CALL and event.tool_name == "handoff":
        event.data["handoff"] = event.tool_args.get("target_agent")
        event.data["abort"] = True
        event.data["abort_reason"] = f"handoff to {event.data['handoff']}"
    return event
