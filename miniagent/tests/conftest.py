"""pytest configuration for the miniagent test suite.

- Registers the `live` marker (network/LLM tests).
- Ensures the repo root is importable so `import miniagent` works whether
  the package is installed or you're running from a checkout.
"""

import os
import sys

import pytest

# Make `import miniagent` work from a checkout (tests live inside the package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: tests that need a live LLM / network (MINIAGENT_RUN_LIVE=1)")
