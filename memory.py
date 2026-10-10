"""Memory — a bounded list of the recent conversation.

Keep only the last N messages so the context stays small. The system prompt
(the agent's identity) is pinned and never dropped. No token counting — a
simple message cap is enough for a lean agent.
"""


class Memory:
    """A sliding window over the message list, pinned to the system prompt."""

    def __init__(self, max_messages: int = 50):
        # Sized to hold a full run (10 tool-call rounds ~= 22 messages) so a
        # normal run is never trimmed. The real stop is the tool-call counter
        # in agent.py; this is just a safety net against unbounded growth.
        self.max_messages = max_messages
        self.messages: list = []

    def add(self, msg: dict) -> None:
        self.messages.append(msg)
        self.trim()  # stay within the cap on every add

    def trim(self) -> int:
        """Drop oldest non-system messages until at or under the cap."""
        dropped = 0
        while len(self.messages) > self.max_messages:
            idx = next((i for i, m in enumerate(self.messages)
                        if m.get("role") != "system"), None)
            if idx is None:
                break  # only system messages left; nothing safe to drop
            self.messages.pop(idx)
            dropped += 1
        return dropped

    def get_messages(self) -> list:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []
