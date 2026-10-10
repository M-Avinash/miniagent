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

from .events import HookEvent

# A hook takes an event and returns either a (possibly new) event to keep
# going with, or None to keep the one it was given.
HookFn = Callable[[HookEvent], Optional[HookEvent]]


class HookPipeline:
    """Runs an ordered list of hooks over a HookEvent."""

    def __init__(self, hooks: list = None):
        self.hooks: list = list(hooks) if hooks else []

    def add(self, hook: HookFn) -> "HookPipeline":
        self.hooks.append(hook)
        return self  # allows chaining: pipeline.add(a).add(b)

    def emit(self, event: HookEvent) -> HookEvent:
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
