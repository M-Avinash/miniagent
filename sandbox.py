"""sandbox.py — run Python code in a subprocess.

The run_python tool executes code in a separate process. If the code hangs,
crashes, or calls os._exit(), only the subprocess dies — the agent keeps running.
Errors and timeouts come back as strings, never as exceptions that kill the agent.
"""

import subprocess
import sys

from .tool import tool


@tool
def run_python(code: str, timeout: int = 10) -> str:
    """Run Python code in a subprocess and return its output.

    The code runs in a separate process. stdout and stderr are captured and
    returned. If the code raises an exception, exits non-zero, or times out,
    the error is returned as a string (not raised).

    Args:
        code (str): The Python code to execute.
        timeout (int): Max seconds to wait before killing the subprocess.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] code did not finish within {timeout}s"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

    parts = [f"[exit {proc.returncode}]"]
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(proc.stderr.rstrip())
    return "\n".join(parts)
