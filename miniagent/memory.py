"""Memory — a bounded list of the recent conversation.

Keep only the last N messages so the context stays small. The system prompt
(the agent's identity) is pinned and never dropped. No token counting — a
simple message cap is enough for a lean agent.
"""


class Memory:
    """A sliding window over the message list, pinned to the system prompt.

    max_messages defaults to 50, sized to hold a full run (10 tool-call
    rounds ~= 22 messages) so a normal run is never trimmed. The real stop
    is the tool-call counter in agent.py; this cap is just a safety net
    against unbounded growth.
    """

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: list = []

    def add(self, msg: dict) -> None:
        """Append a message, then trim to stay within the cap."""
        self.messages.append(msg)
        self.trim()

    def trim(self) -> int:
        """Drop oldest non-system messages until at or under the cap.

        Returns the number of messages dropped.
        """
        dropped = 0
        while len(self.messages) > self.max_messages:
            idx = next((i for i, m in enumerate(self.messages)
                        if m.get("role") != "system"), None)
            if idx is None:
                break
            self.messages.pop(idx)
            dropped += 1
        return dropped

    def get_messages(self) -> list:
        """Return a copy of the current message list."""
        return list(self.messages)

    def clear(self) -> None:
        """Drop all messages (start a fresh conversation)."""
        self.messages = []
