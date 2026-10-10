"""Tool system — turn Python functions into LLM-callable tools.

A function with type hints and a docstring becomes a fully-specified tool:
    - type hints  -> JSON schema (what shapes the LLM may send)
    - docstring   -> descriptions (what the parameters mean)
    - defaults    -> which parameters are optional
"""

import asyncio
import inspect
import json
import re
from typing import get_type_hints

# Python type -> JSON Schema type.
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp):
    """Map a Python type to a JSON schema fragment."""
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}
    # Generic types like list[str]
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        args = getattr(tp, "__args__", None)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}
    return {"type": "string"}


def _parse_docstring(docstring):
    """Parse a docstring into (description, param_descriptions).

    Convention:
        First line is the description.
        An "Args:" section with "name (type): description" lines.
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description = lines[0].strip()
    params = {}
    in_args = False

    for line in lines[1:]:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", stripped)
            if m:
                params[m.group(1)] = m.group(2).strip()
            elif stripped:
                in_args = False  # left the Args section

    return description, params


def _build_tool_schema(func):
    """Build an OpenAI-compatible tool definition from a Python function."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc_description, param_docs = _parse_docstring(func.__doc__)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        tp = hints.get(name, str)
        schema = _python_type_to_json_schema(tp)
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc_description or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class Tool:
    """A callable tool the LLM can invoke: a function plus its schema."""

    def __init__(self, func, name=None, description=None):
        self.func = func
        self.name = name or func.__name__
        self.is_async = inspect.iscoroutinefunction(func)
        self.schema = _build_tool_schema(func)
        if name:
            self.schema["function"]["name"] = name
        if description:
            self.schema["function"]["description"] = description

    def to_openai_schema(self):
        """The JSON schema sent to the LLM."""
        return self.schema

    async def execute(self, **kwargs):
        """Run the tool. Always returns a string (what goes back to the LLM)."""
        if self.is_async:
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def __call__(self, **kwargs):
        """Call the tool directly (for your code, not the LLM)."""
        try:
            asyncio.get_running_loop()
            # Already inside async code — can't asyncio.run() here.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.execute(**kwargs)).result()
        except RuntimeError:
            return asyncio.run(self.execute(**kwargs))


def tool(func=None, *, name=None, description=None):
    """Use as @tool or @tool(name="...", description="...")."""
    if func is not None:
        return Tool(func, name=name, description=description)

    def wrapper(f):
        return Tool(f, name=name, description=description)
    return wrapper
