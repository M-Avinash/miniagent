"""observation.py — distilled observational memory.

ObservationalMemory distills each agent run into a brief observation
(1-3 lines, ~50 words) and appends it to a capped JSONL file. On the next
run, the last N observations are injected into the system prompt so the
agent has continuity without the full transcript.

Usage:
    from miniagent.observation import ObservationalMemory
    om = ObservationalMemory("my-agent", path="observations.jsonl", max_entries=50)
    agent = Agent("...", hooks=[om.hook])
    # On next run:
    system_prompt = base_prompt + om.recall_prompt()
"""

import json
import os
from collections import deque


class ObservationalMemory:
    """A rolling log of distilled observations, stored as JSONL.

    Args:
        label: A short tag identifying this agent (stored on each observation).
        path: Path to the JSONL file.
        max_entries: Cap on the number of observations kept (FIFO).
        recall_count: How many recent observations to recall into the prompt.
    """

    def __init__(self, label: str, path: str = "observations.jsonl",
                 max_entries: int = 50, recall_count: int = 5):
        self.label = label
        self.path = path
        self.max_entries = max_entries
        self.recall_count = recall_count
        self._entries = deque(maxlen=max_entries)
        self._load()

    def _load(self):
        """Load existing observations from the JSONL file."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self):
        """Persist all observations to the JSONL file (overwrite, capped)."""
        try:
            with open(self.path, "w") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def hook(self, event):
        """AGENT_END hook: distill the run into an observation and store it."""
        if event.type.value != "agent_end":
            return event

        messages = event.messages or []
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")[:100]
                break
        assistant_msg = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                assistant_msg = m.get("content", "")[:200]
                break

        observation = {
            "label": self.label,
            "user": user_msg,
            "assistant": assistant_msg,
            "turns": event.turn,
        }
        self._entries.append(observation)
        self._save()
        return event

    def recall(self, count: int = None):
        """Return the last N observations (most recent last)."""
        count = count or self.recall_count
        entries = list(self._entries)
        return entries[-count:]

    def recall_prompt(self) -> str:
        """Build a system-prompt suffix with recent observations."""
        recent = self.recall()
        if not recent:
            return ""
        lines = ["\n\n## Recent observations (from previous runs)"]
        for obs in recent:
            lines.append(f"- User: {obs.get('user', '?')}")
            lines.append(f"  Agent: {obs.get('assistant', '?')[:100]}")
        return "\n".join(lines)
