"""heartbeat.py — a proactive agent awakener (OpenClaw-style heartbeat).

A HeartbeatAgent wakes an agent on an interval with a heartbeat prompt and
lets the agent decide whether anything needs attention. This is what turns a
reactive chatbot into a proactive agent that works while you are away.

Two modes:

1. Proactive awakener (default): the heartbeat sends a "check for pending
   work" prompt. If the agent finds nothing to do, it replies HEARTBEAT_OK
   and the beat is a no-op (no output surfaced). This mirrors OpenClaw's
   heartbeat contract.

2. Maker-checker (optional): pass a checker agent. The heartbeat keeps
   beating until the checker reports the task is done.

It is NOT a new loop — it is a plain while-loop with a sleep that keeps
invoking the existing agent loop (agent.py). The real agent logic still lives
in agent.py; the heartbeat only decides WHEN to run it and whether the result
is a no-op.
"""

import time

from .structured import parse_structured

HEARTBEAT_OK = "HEARTBEAT_OK"

DEFAULT_HEARTBEAT_PROMPT = (
    "You are running a periodic heartbeat check. Review the current state and "
    "do anything that genuinely needs attention right now. If nothing needs "
    "attention, reply with exactly: " + HEARTBEAT_OK
)


class HeartbeatAgent:
    """Wakes an agent on an interval; the agent decides if action is needed.

    Args:
        agent: the agent to wake each beat.
        prompt: the heartbeat prompt. Defaults to a proactive "check for
            pending work" prompt. Pass a fixed task to re-run the same work.
        interval: seconds between beats.
        checker: optional Agent that reviews each beat's output. When set,
            the heartbeat stops once the checker reports done (maker-checker).
        max_beats: optional hard cap on the number of beats (None = unlimited).
        ack: the no-op token. If the agent's reply is (only) this, the beat is
            a no-op. Defaults to HEARTBEAT_OK.
        on_alert: optional callback invoked with the AgentResult whenever a
            beat is NOT a no-op (i.e. the agent surfaced something). Use this
            to deliver alerts (print, log, webhook, etc.).
    """

    def __init__(self, agent, prompt=None, interval=30, checker=None,
                 max_beats=None, ack=HEARTBEAT_OK, on_alert=None):
        self.agent = agent
        self.prompt = prompt if prompt is not None else DEFAULT_HEARTBEAT_PROMPT
        self.interval = interval
        self.checker = checker
        self.max_beats = max_beats
        self.ack = ack
        self.on_alert = on_alert
        self._stop = False
        self.beat = 0
        self.last_result = None
        self.last_was_ack = False

    def stop(self):
        """Signal run_forever to stop after the current beat."""
        self._stop = True

    def run_once(self):
        """Run a single beat. Returns the agent's AgentResult.

        Sets last_was_ack to True if the agent replied with the no-op token.
        Fires on_alert(result) when the beat is not a no-op.
        """
        self.beat += 1
        self.last_result = self.agent.run(self.prompt)
        self.last_was_ack = self._is_ack(self.last_result.text)
        if not self.last_was_ack and self.on_alert is not None:
            self.on_alert(self.last_result)
        return self.last_result

    def run_forever(self):
        """Run beats until stop(), the checker says done, or max_beats is hit."""
        while not self._stop:
            result = self.run_once()

            if self.checker is not None and self._check(result):
                break

            if self.max_beats is not None and self.beat >= self.max_beats:
                break

            time.sleep(self.interval)
        return self.last_result

    def run(self, n_times=5):
        """Run a bounded number of beats (no checker needed)."""
        for _ in range(n_times):
            if self._stop:
                break
            self.run_once()
            time.sleep(self.interval)
        return self.last_result

    def _is_ack(self, text):
        """True if the reply is the no-op token (ignoring surrounding chatter)."""
        if not text:
            return False
        cleaned = text.strip().strip(".*").strip()
        return self.ack in cleaned and len(cleaned) <= len(self.ack) + 20

    def _check(self, result):
        """Ask the checker agent if the maker's output is 'done'. Returns bool.

        A checker that crashes is treated as 'not done' (returns False), so a
        faulty checker never stops the heartbeat.
        """
        prompt = (
            "You are a checker. Review the following output and decide if the "
            "task is complete and correct. Respond with ONLY a JSON object: "
            '{"done": true} or {"done": false, "reason": "what is missing"}.\n\n'
            f"Task: {self.prompt}\n\nOutput to review:\n{result.text}"
        )
        try:
            check = self.checker.run(prompt)
        except Exception:
            return False
        parsed = parse_structured(check.text, {"done": bool})
        return bool(parsed.get("done", False)) if isinstance(parsed, dict) else False
