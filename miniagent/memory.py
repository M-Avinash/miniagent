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

        Tool-call safe: an assistant message that carries tool_calls is
        dropped together with the tool messages that answer it, so a `tool`
        message is never orphaned from its `tool_call_id` (OpenAI-compatible
        APIs reject such a message). Returns the number of messages dropped.
        """
        dropped = 0
        while len(self.messages) > self.max_messages:
            idx = self._next_droppable_index()
            if idx is None:
                break
            count = self._drop_block(idx)
            self.messages = self.messages[:idx] + self.messages[idx + count:]
            dropped += count
        return dropped

    def _next_droppable_index(self):
        """Index of the oldest non-system message, or None if only system
        messages remain (nothing safe to drop)."""
        for i, m in enumerate(self.messages):
            if m.get("role") != "system":
                return i
        return None

    def _drop_block(self, idx: int) -> int:
        """How many consecutive messages starting at idx must be dropped
        together to keep tool_calls/tool pairs intact.

        If the message at idx is an assistant message with tool_calls, the
        tool messages that follow it (answering those calls) must go with it.
        """
        msg = self.messages[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pending = {tc["id"] for tc in msg["tool_calls"] if tc.get("id")}
            count = 1
            while pending and idx + count < len(self.messages):
                nxt = self.messages[idx + count]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in pending:
                    pending.discard(nxt["tool_call_id"])
                    count += 1
                else:
                    break
            return count
        return 1

    def get_messages(self) -> list:
        """Return a copy of the current message list."""
        return list(self.messages)

    def clear(self) -> None:
        """Drop all messages (start a fresh conversation)."""
        self.messages = []
