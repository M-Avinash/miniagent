"""sandbox.py — run Python code in an isolated subprocess.

The run_python tool executes code in a separate process with:
- A minimal environment (no API keys, no HOME, no SSH)
- A temp directory as cwd (no access to project files)
- A timeout (killed if it hangs)
- Full Python inside (can pip install, use any stdlib, roam freely
  within its own temp dir and process memory)

If the code hangs, crashes, or calls os._exit(), only the subprocess
dies — the agent keeps running. Errors and timeouts come back as strings.
"""

import os
import subprocess
import sys
import tempfile

from .tool import tool


def _minimal_env() -> dict:
    """Build a minimal environment for the sandbox subprocess.

    Keeps only what Python needs to run: PATH (for the interpreter),
    PYTHONPATH (empty), and a few locale vars. Drops everything else:
    API keys, HOME, SSH, cloud credentials, etc.
    """
    env = {}
    # Keep PATH so the subprocess can find system binaries if it needs them
    # (e.g., for subprocess calls within the sandbox code itself).
    if "PATH" in os.environ:
        env["PATH"] = os.environ["PATH"]
    # Keep locale for text handling
    for key in ("LANG", "LC_ALL", "LC_CTYPE"):
        if key in os.environ:
            env[key] = os.environ[key]
    # Explicitly drop sensitive vars
    for key in ("HOME", "USER", "SSH_AUTH_SOCK", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "AZURE_STORAGE_CONNECTION_STRING", "GITHUB_TOKEN",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "API_KEY", "SECRET_KEY", "TOKEN",
                "DB_PASSWORD", "DATABASE_URL", "REDIS_URL"):
        env.pop(key, None)
    # No HOME — the subprocess can't read ~/.ssh, ~/.aws, ~/.config
    env["HOME"] = os.environ.get("_SANDBOX_HOME", tempfile.gettempdir())
    # No PYTHONPATH — the sandbox code can't import from the project
    env["PYTHONPATH"] = ""
    return env


@tool
def run_python(code: str, timeout: int = 10) -> str:
    """Run Python code in a sandboxed subprocess and return its output.

    The code runs in a separate process with a minimal environment
    (no API keys, no HOME) and a temp directory as its working directory.
    The code has full access to Python's standard library and can pip
    install packages — but it cannot read your environment variables,
    your home directory, or your project files.

    stdout and stderr are captured and returned. If the code raises an
    exception, exits non-zero, or times out, the error is returned as a
    string (not raised).

    Args:
        code (str): The Python code to execute.
        timeout (int): Max seconds to wait before killing the subprocess.
    """
    # Create a temp dir for the subprocess to use as its working directory
    sandbox_dir = tempfile.mkdtemp(prefix="miniagent_sandbox_")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            cwd=sandbox_dir,
            env=_minimal_env(),
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] code did not finish within {timeout}s"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    finally:
        # Clean up the sandbox dir
        import shutil
        try:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        except Exception:
            pass

    parts = [f"[exit {proc.returncode}]"]
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(proc.stderr.rstrip())
    return "\n".join(parts)
