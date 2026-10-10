"""observe.py — verbose logging hook.

Attach verbose_log to an Agent's hooks to see every event as it happens:
turns, LLM calls, tool calls, tool results, token usage. Per-agent labeling
so multiple agents in the same process don't get mixed up.

Usage:
    from miniagent.observe import verbose_log
    agent = Agent("...", hooks=[verbose_log("my-agent")])
"""

import json
import sys
import time


def verbose_log(label: str = "agent", debug: bool = False, log_file: str = None):
    """Return a hook that logs every agent event.

    Args:
        label: Short name to prefix each log line (e.g. "researcher").
        debug: If True, print full message payloads and tool results.
        log_file: If set, also append each log line to this file (JSONL).
    """
    def _log(event):
        t = time.strftime("%H:%M:%S")
        prefix = f"[{t}] [{label}]"
        line = None

        if event.type.value == "agent_start":
            msg = event.data.get("user_message", "")
            line = f"{prefix} START: {msg[:120]}"
        elif event.type.value == "llm_call":
            n_msgs = len(event.messages or [])
            line = f"{prefix} LLM call (turn {event.turn}, {n_msgs} messages)"
        elif event.type.value == "llm_response":
            msg = event.response
            if msg is not None:
                content = (msg.content or "")[:80]
                n_tools = len(msg.tool_calls or [])
                line = f"{prefix} LLM response (turn {event.turn}): {content!r}"
                if n_tools:
                    line += f" + {n_tools} tool call(s)"
        elif event.type.value == "tool_call":
            line = f"{prefix} TOOL: {event.tool_name}({event.tool_args})"
        elif event.type.value == "tool_result":
            result = event.tool_result or ""
            line = f"{prefix} RESULT: {result[:120]}"
            if debug and len(result) > 120:
                line += f" ... ({len(result)} chars total)"
        elif event.type.value == "agent_end":
            line = f"{prefix} END (turn {event.turn})"
        elif event.type.value == "error":
            line = f"{prefix} ERROR: {event.error}"

        if line:
            print(line, file=sys.stderr, flush=True)
            if log_file:
                try:
                    with open(log_file, "a") as f:
                        f.write(json.dumps({"label": label, "type": event.type.value,
                                            "turn": event.turn, "log": line}) + "\n")
                except OSError:
                    pass   # logging must never crash the agent

        # Debug mode: dump full payloads
        if debug:
            if event.type.value == "llm_call" and event.messages:
                for m in event.messages:
                    print(f"{prefix}   msg: {json.dumps(m, default=str)[:200]}",
                          file=sys.stderr, flush=True)
            if event.type.value == "tool_result" and event.tool_result:
                print(f"{prefix}   full result: {event.tool_result}", file=sys.stderr, flush=True)

        return event

    return _log
