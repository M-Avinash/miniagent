"""observe.py — logging and tracing hooks.

Two hooks:
  verbose_log  — human-readable terminal log (per-agent labeled)
  trace_hook   — machine-readable JSONL audit trail (one line per event)

Usage:
    from miniagent.observe import verbose_log, trace_hook
    agent = Agent("...", hooks=[verbose_log("my-agent"), trace_hook("trace.jsonl")])
"""

import json
import sys
import time
import uuid


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
            line = f"{prefix} START: {msg[:200]}"
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
                    pass

        if debug:
            if event.type.value == "llm_call" and event.messages:
                for m in event.messages:
                    print(f"{prefix}   msg: {json.dumps(m, default=str)[:200]}",
                          file=sys.stderr, flush=True)
            if event.type.value == "tool_result" and event.tool_result:
                print(f"{prefix}   full result: {event.tool_result}", file=sys.stderr, flush=True)

        return event

    return _log


def trace_hook(path: str, run_id: str = None):
    """Return a hook that writes a machine-readable JSONL audit trail.

    One line per event, timestamped, with agent name, turn number, and the
    full payload. Use for production debugging, replay, or metrics.

    Args:
        path: File to append JSONL lines to.
        run_id: Optional run identifier (auto-generated if not provided).
            Use the same run_id across multiple agents in a multi-agent run
            to correlate their events.

    Example:
        run_id = str(uuid.uuid4())
        agent_a = Agent("...", hooks=[trace_hook("trace.jsonl", run_id)])
        agent_b = Agent("...", hooks=[trace_hook("trace.jsonl", run_id)])
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    def _trace(event):
        entry = {
            "ts": time.time(),
            "run_id": run_id,
            "event": event.type.value,
            "turn": event.turn,
        }
        if event.tool_name:
            entry["tool_name"] = event.tool_name
        if event.tool_args is not None:
            entry["tool_args"] = event.tool_args
        if event.tool_result is not None:
            entry["tool_result"] = event.tool_result
        if event.data:
            entry["data"] = event.data
        if event.error:
            entry["error"] = str(event.error)
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass
        return event

    return _trace
