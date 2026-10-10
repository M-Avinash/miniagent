"""structured.py — get clean, typed output from the LLM.

No Pydantic. Three small functions:
  build_structured_prompt  -> a system prompt that demands JSON-only output
  extract_json             -> pull a JSON object/array out of messy model text
  parse_structured         -> extract_json + json.loads (+ optional key check)
"""

import json
import re


def build_structured_prompt(schema: dict, instruction: str) -> str:
    """Build a system prompt that asks the model to return JSON matching `schema`.

    `schema` is a plain dict describing the expected shape, e.g.:
        {"type": "object", "properties": {"city": {"type": "string"}}}
    """
    example = json.dumps(schema, indent=2)
    return (
        f"{instruction}\n\n"
        "Respond with ONLY a valid JSON value that matches this schema. "
        "No prose, no markdown, no code fences — just the JSON.\n"
        f"Schema:\n{example}"
    )


def extract_json(text: str):
    """Pull the first JSON object or array out of possibly-messy model text.

    Tries, in order: the whole string, a ```json fenced block, then the first
    balanced {...} or [...] span. Returns the parsed value, or raises.
    """
    if text is None:
        raise ValueError("no text to parse")
    text = text.strip()

    # 1. The whole thing is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. A fenced block: ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. The first balanced object or array anywhere in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no valid JSON found in text: {text[:200]!r}")


def parse_structured(text: str, fields: list = None):
    """Extract + parse JSON from model text. Optionally check required keys.

    `fields` is a list of expected top-level keys; if given and any are missing,
    raises ValueError so the caller can retry.
    """
    value = extract_json(text)
    if fields and isinstance(value, dict):
        missing = [f for f in fields if f not in value]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
    return value
