"""heartbeat.py — run an agent on an interval.

A HeartbeatAgent wraps an Agent and calls agent.run(task) every `interval`
seconds. It is NOT a new kind of loop — it's a plain while-loop with a sleep
that keeps invoking the existing agent loop (agent.py). The real agent logic
still lives in agent.py; the heartbeat only decides WHEN to run it.

Optional maker-checker: pass a second agent (the checker) that reviews each
beat's output. The heartbeat keeps going until the checker says "done."
"""

import time

from .structured import parse_structured


class HeartbeatAgent:
    def __init__(self, agent, task, interval=30, checker=None, max_beats=None):
        self.agent = agent            # the maker — runs the task each beat
        self.task = task              # the prompt sent to the maker each beat
        self.interval = interval      # seconds between beats
        self.checker = checker        # optional Agent that validates the maker's output
        self.max_beats = max_beats    # optional hard cap on number of beats (None = unlimited)
        self._stop = False
        self.beat = 0
        self.last_result = None

    # ------------------------------------------------------------- control
    def stop(self):
        """Signal run_forever to stop after the current beat."""
        self._stop = True

    # ------------------------------------------------------------- running
    def run_once(self):
        """Run a single beat. Returns the maker's AgentResult."""
        self.beat += 1
        self.last_result = self.agent.run(self.task)
        return self.last_result

    def run_forever(self):
        """Run beats until stop() is called, the checker says done, or max_beats is hit."""
        while not self._stop:
            result = self.run_once()

            # Optional maker-checker: stop when the checker is satisfied.
            if self.checker is not None:
                verdict = self._check(result)
                if verdict:
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

    # ------------------------------------------------------------- checking
    def _check(self, result):
        """Ask the checker agent if the maker's output is 'done'. Returns bool."""
        prompt = (
            "You are a checker. Review the following output and decide if the task is "
            "complete and correct. Respond with ONLY a JSON object: "
            '{"done": true} or {"done": false, "reason": "what is missing"}.\n\n'
            f"Task: {self.task}\n\nOutput to review:\n{result.text}"
        )
        try:
            check = self.checker.run(prompt)
        except Exception:
            return False   # a checker that crashes shouldn't stop the heartbeat
        parsed = parse_structured(check.text, {"done": bool})
        return bool(parsed.get("done", False)) if isinstance(parsed, dict) else False
