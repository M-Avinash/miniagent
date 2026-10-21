"""agent.py — the agent loop. One source of truth for how an agent runs.

An Agent owns an LLM, a system prompt, a set of tools, a Memory, and a hook
pipeline. run() drives the unified tool-calling loop:

  [system] [user] -> LLM -> (tool_call -> tool result -> LLM)* -> final answer

Tools stay available every turn. Turn 1 uses tool_choice="required" (so the
model must use a tool rather than answer from its weights); later turns use
"auto" so the model can stop and answer. The loop is bounded by max_tool_calls.
"""

import dataclasses
import json
import typing

from .events import AgentResult, EventType, HookEvent
from .hooks import HookPipeline
from .llm import LLMClient
from .memory import Memory
from .orchestrator import make_handoff_marker, make_abort_marker
from .structured import build_structured_prompt, extract_json
from .tool import tool


class Agent:
    """A single agent: an LLM + tools + memory + hooks, driven by run()."""

    def __init__(self, system_prompt, llm=None, tools=None, memory=None,
                 hooks=None, max_tool_calls=10, max_messages=50, name=None,
                 stream=False, stream_hook=None, output_type=None):
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools = tools or []
        self.memory = memory or Memory(max_messages=max_messages)
        self.hooks = HookPipeline(hooks)
        self.max_tool_calls = max_tool_calls
        self.name = name
        self.stream = stream
        self.stream_hook = stream_hook  # callable(delta_text, accumulated_msg, turn)
        self.output_type = output_type
        self._output_schema = self._build_output_schema(output_type) if output_type else None
        if output_type:
            self.system_prompt = build_structured_prompt(
                self._output_schema, self.system_prompt
            )

    def run(self, user_message: str) -> AgentResult:
        """Run the tool-calling loop for one user message.

        Each run starts with a fresh conversation (system prompt + user
        message). The loop is bounded by max_tool_calls; if it runs out of
        turns without a final answer, it returns a graceful abort. A hook can
        set event.data["abort"] to stop early, or event.data["handoff"] to
        hand off to another agent (multi-agent).

        If stream=True, the LLM response is consumed token-by-token. The
        stream_hook (if set) is called with (delta_text, accumulated_msg,
        turn) for each token. The final AgentResult is identical to the
        non-streaming path.
        """
        self.memory.clear()
        self.memory.add({"role": "system", "content": self.system_prompt})
        self.memory.add({"role": "user", "content": user_message})

        all_tool_calls = []
        usage = {}
        self._tool_cache = {}  # per-run cache: (tool_name, args_hash) -> result

        ev = self.hooks.emit(HookEvent(type=EventType.AGENT_START, data={"user_message": user_message}))
        if ev.data.get("abort"):
            return self._aborted("aborted at start: " + str(ev.data.get("abort_reason", "")),
                                 0, all_tool_calls, usage)

        max_turns = self.max_tool_calls + 1
        last_text = ""

        for turn in range(1, max_turns + 1):
            messages = self.memory.get_messages()
            ev = self.hooks.emit(HookEvent(type=EventType.LLM_CALL, turn=turn, messages=messages))
            if ev.data.get("abort"):
                return self._aborted("aborted: " + str(ev.data.get("abort_reason", "")),
                                     turn, all_tool_calls, usage)
            messages = ev.messages if ev.messages is not None else messages

            tool_choice = "required" if (turn == 1 and self.tools) else "auto"
            msg, turn_usage = self._llm_call(messages, tool_choice, turn)
            if turn_usage:
                usage = self._merge_usage(usage, turn_usage)

            ev = self.hooks.emit(HookEvent(type=EventType.LLM_RESPONSE, turn=turn,
                                           messages=messages, response=msg))
            if ev.data.get("abort"):
                return self._aborted("aborted: " + str(ev.data.get("abort_reason", "")),
                                     turn, all_tool_calls, usage)
            msg = ev.response if ev.response is not None else msg

            # msg is either a pydantic object (non-stream) or a dict (stream)
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            else:
                msg_dict = msg
            self.memory.add(msg_dict)
            last_text = msg_dict.get("content") or ""

            if not msg_dict.get("tool_calls"):
                ev = HookEvent(type=EventType.AGENT_END, turn=turn,
                               messages=self.memory.get_messages())
                ev.data["final_text"] = last_text
                ev = self.hooks.emit(ev)
                final_text = ev.data.get("final_text", last_text) if ev.data else last_text
                handoff = ev.data.get("handoff")
                structured = self._parse_structured_output(final_text)
                return AgentResult(text=final_text, turns=turn, tool_calls=all_tool_calls,
                                   usage=usage, handoff=handoff, structured=structured)

            for tc in msg_dict["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                all_tool_calls.append({"name": name, "args": args})
                result, tool_ev = self._run_tool_dict(name, tc["id"], turn, args)
                self.memory.add({"role": "tool", "tool_call_id": tc["id"], "content": result})

                if tool_ev and tool_ev.data.get("handoff"):
                    return AgentResult(
                        text=last_text or make_handoff_marker(tool_ev.data['handoff']),
                        turns=turn, tool_calls=all_tool_calls,
                        usage=usage, handoff=tool_ev.data["handoff"])

        return self._aborted(f"max tool calls ({self.max_tool_calls}) reached",
                             max_turns, all_tool_calls, usage, text=last_text)

    def _llm_call(self, messages, tool_choice, turn):
        """Make one LLM call. Returns a message object (pydantic) or dict (stream).

        In streaming mode, yields tokens to self.stream_hook and returns the
        final accumulated message as a plain dict.
        """
        schemas = [t.to_openai_schema() for t in self.tools] if self.tools else None

        if self.stream:
            kwargs = {"stream": True}
            gen = self.llm.chat(messages, tools=schemas, tool_choice=tool_choice, **kwargs)
            final_msg = None
            for delta, acc_msg in gen:
                final_msg = acc_msg
                if self.stream_hook and delta and delta.content:
                    self.stream_hook(delta.content, acc_msg, turn)
            return (final_msg or {"role": "assistant", "content": None, "tool_calls": None}), {}
        else:
            resp = self.llm.chat(messages, tools=schemas, tool_choice=tool_choice)
            turn_usage = self._usage_dict(resp.usage) if getattr(resp, "usage", None) else {}
            return resp.choices[0].message, turn_usage

    def _run_tool_dict(self, name, tool_call_id, turn, args):
        """Run a tool call from a dict-format message (streaming path).

        Returns (result_string, tool_call_event).
        """
        ev = self.hooks.emit(HookEvent(type=EventType.TOOL_CALL, turn=turn,
                                       tool_name=name, tool_args=args))
        if ev.data.get("abort"):
            return make_abort_marker(ev.data.get('abort_reason', 'tool call blocked')), ev
        args = ev.tool_args if ev.tool_args is not None else args

        # Validate args against tool schema
        tool_obj = self._find_tool(name)
        if tool_obj is None:
            return f"[error: unknown tool '{name}']", ev

        validation_error = self._validate_tool_args(tool_obj, args)
        if validation_error:
            return f"[error: {validation_error}]", ev

        # Per-run cache: same tool + same args → same result (no re-execution)
        cache_key = (name, json.dumps(args, sort_keys=True, default=str))
        if cache_key in self._tool_cache:
            result = self._tool_cache[cache_key]
        else:
            try:
                result = tool_obj(args)
            except Exception as e:
                result = f"[error: {type(e).__name__}: {e}]"
            self._tool_cache[cache_key] = result

        if isinstance(result, (dict, list)):
            result = json.dumps(result)

        ev_result = self.hooks.emit(HookEvent(type=EventType.TOOL_RESULT, turn=turn,
                                              tool_name=name, tool_args=args, tool_result=result))
        final_result = ev_result.tool_result if ev_result.tool_result is not None else result
        return final_result, ev

    @staticmethod
    def _validate_tool_args(tool_obj, args: dict) -> str:
        """Validate tool args against the generated schema.

        Returns an error string if validation fails, or empty string if OK.
        Checks: missing required args, wrong types, unknown args.
        """
        schema = tool_obj.schema["function"]["parameters"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Missing required args
        missing = [r for r in required if r not in args]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"

        # Wrong types
        type_checks = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "object": (dict,),
        }
        for key, value in args.items():
            if key not in properties:
                continue  # unknown args are ignored (lenient mode)
            expected_type = properties[key].get("type")
            if expected_type and expected_type in type_checks:
                allowed = type_checks[expected_type]
                # bool is a subclass of int — reject bool where int expected
                if expected_type == "integer" and isinstance(value, bool):
                    return f"argument '{key}' expected {expected_type}, got boolean"
                if not isinstance(value, allowed):
                    return (f"argument '{key}' expected {expected_type}, "
                            f"got {type(value).__name__}")

        return ""

    def _find_tool(self, name: str):
        """Return the Tool with the given name, or None."""
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _parse_structured_output(self, text: str):
        """Parse the final text against output_type. Returns the typed object
        or None if parsing fails (text is still available in result.text)."""
        if not self.output_type:
            return None
        try:
            raw = extract_json(text)
            if dataclasses.is_dataclass(self.output_type):
                return self.output_type(**raw)
            elif hasattr(self.output_type, 'model_validate'):
                return self.output_type.model_validate(raw)
            elif isinstance(self.output_type, type) and issubclass(self.output_type, dict):
                return raw
            else:
                return raw
        except (ValueError, TypeError, KeyError) as e:
            return None

    @staticmethod
    def _build_output_schema(output_type) -> dict:
        """Build a JSON schema dict from a dataclass or pydantic model.

        Uses typing.get_type_hints (no eval) to resolve forward references
        safely.
        """
        if dataclasses.is_dataclass(output_type):
            try:
                hints = typing.get_type_hints(output_type)
            except (NameError, TypeError):
                hints = Agent._resolve_hints_fallback(output_type)
            type_map = {
                str: "string", int: "integer", float: "number",
                bool: "boolean", list: "array", dict: "object",
            }
            props = {}
            required = []
            for f in dataclasses.fields(output_type):
                py_type = hints.get(f.name, str)
                # Handle generic aliases like list[str], dict[str, int]
                origin = getattr(py_type, "__origin__", None)
                if origin is not None:
                    py_type = origin
                json_type = type_map.get(py_type, "string")
                props[f.name] = {"type": json_type}
                if f.default is dataclasses.MISSING:
                    required.append(f.name)
            return {"type": "object", "properties": props, "required": required}
        elif hasattr(output_type, 'model_json_schema'):
            return output_type.model_json_schema()
        else:
            return {"type": "object"}

    @staticmethod
    def _resolve_hints_fallback(output_type) -> dict:
        """Resolve dataclass field types without eval.

        Handles simple forward refs like 'list[str]', 'dict', 'int' by
        matching against a small table. Unresolvable types default to str.
        """
        import re
        simple = {
            "str": str, "int": int, "float": float, "bool": bool,
            "list": list, "dict": dict,
        }
        hints = {}
        for f in dataclasses.fields(output_type):
            t = f.type
            if not isinstance(t, str):
                hints[f.name] = t
                continue
            t = t.strip()
            if t in simple:
                hints[f.name] = simple[t]
            else:
                m = re.match(r"^(list|dict)\s*\[", t)
                if m:
                    hints[f.name] = simple[m.group(1)]
                else:
                    hints[f.name] = str
        return hints

    @staticmethod
    def _aborted(reason, turns, tool_calls, usage, text="") -> AgentResult:
        """Build an AgentResult marked as aborted."""
        return AgentResult(text=text, turns=turns, tool_calls=tool_calls,
                           usage=usage, aborted=True, abort_reason=reason)

    @staticmethod
    def _usage_dict(usage) -> dict:
        """Flatten an OpenAI usage object into a plain dict."""
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    @staticmethod
    def _merge_usage(acc: dict, turn: dict) -> dict:
        """Sum per-turn usage into the running total (thread-safe: no shared
        instance state)."""
        merged = dict(acc)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            merged[k] = merged.get(k, 0) + turn.get(k, 0)
        return merged


# ---------------------------------------------------------------------------
# Supervisor Pattern (#12)
# ---------------------------------------------------------------------------

def make_supervisor_tool(agents: dict, description: str = None):
    """Create a tool that lets a supervisor agent call worker agents.

    The supervisor's loop IS the orchestration. It calls workers, reads
    results, decides next steps. No separate orchestrator needed.

    Args:
        agents: Dict mapping agent name → Agent object.
        description: Optional tool description override.

    Returns:
        A Tool object. The supervisor calls call_worker(agent_name, task)
        and gets back the worker's final answer text.

    Example:
        workers = {
            "researcher": Agent("You research topics...", name="researcher"),
            "writer": Agent("You write summaries...", name="writer"),
        }
        supervisor = Agent(
            "You coordinate workers to complete tasks.",
            tools=[make_supervisor_tool(workers)],
            name="supervisor",
        )
        result = supervisor.run("Research Python asyncio and write a summary")
    """
    valid_names = list(agents.keys())
    names_str = ", ".join(f"'{n}'" for n in valid_names)

    if not description:
        description = f"Call a worker agent. Valid workers: {names_str}."

    @tool(name="call_worker", description=description)
    def call_worker(agent_name: str, task: str) -> str:
        """Call a worker agent with a task.

        Args:
            agent_name (str): Which worker to call.
            task (str): The task to give it.
        """
        if agent_name not in agents:
            return f"[error: unknown worker '{agent_name}'. Valid: {names_str}]"
        try:
            result = agents[agent_name].run(task)
            if result.aborted:
                return f"[error: {agent_name} aborted: {result.abort_reason}]"
            return result.text
        except Exception as e:
            return f"[error: {agent_name} failed: {type(e).__name__}: {e}]"

    return call_worker


# ---------------------------------------------------------------------------
# Agent Composition (#9) — wrap an agent as a tool
# ---------------------------------------------------------------------------

def make_agent_tool(sub_agent: "Agent", description: str = None) -> "Tool":
    """Wrap an agent as a tool so a parent agent can call it.

    The parent's loop IS the hierarchy — it calls the sub-agent like any
    other tool, reads the result, decides next steps. No new orchestration
    layer needed.

    Args:
        sub_agent: The Agent to wrap.
        description: Optional tool description override.

    Returns:
        A Tool object. The parent calls call_<name>(task) and gets back the
        sub-agent's final answer text.

    Example:
        researcher = Agent("You research topics...", name="researcher")
        manager = Agent(
            "You delegate research to the researcher agent.",
            tools=[make_agent_tool(researcher)],
            name="manager",
        )
        result = manager.run("Research the history of the Roman Empire")
    """
    name = sub_agent.name or "sub_agent"
    if not description:
        description = f"Run the {name} agent on a task."

    @tool(name=f"call_{name}", description=description)
    def call_sub_agent(task: str) -> str:
        """Give a task to the sub-agent and get its answer.

        Args:
            task (str): The task to perform.
        """
        try:
            result = sub_agent.run(task)
            if result.aborted:
                return f"[error: {name} aborted: {result.abort_reason}]"
            return result.text
        except Exception as e:
            return f"[error: {name} failed: {type(e).__name__}: {e}]"

    return call_sub_agent
