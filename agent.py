"""agent.py — the agent loop. One source of truth for how an agent runs.

An Agent owns an LLM, a system prompt, a set of tools, a Memory, and a hook
pipeline. run() drives the unified tool-calling loop:

  [system] [user] -> LLM -> (tool_call -> tool result -> LLM)* -> final answer

Tools stay available every turn. Turn 1 uses tool_choice="required" (so the
model must use a tool rather than answer from its weights); later turns use
"auto" so the model can stop and answer. The loop is bounded by max_tool_calls.
"""

import json

from .events import AgentResult, EventType, HookEvent
from .hooks import HookPipeline
from .llm import LLMClient
from .memory import Memory


class Agent:
    def __init__(self, system_prompt, llm=None, tools=None, memory=None,
                 hooks=None, max_tool_calls=10, max_messages=50, name=None):
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools = tools or []
        self.memory = memory or Memory(max_messages=max_messages)
        self.hooks = HookPipeline(hooks)
        self.max_tool_calls = max_tool_calls
        self.name = name   # used for handoff targeting in multi-agent setups

    # ------------------------------------------------------------------ run
    def run(self, user_message: str) -> AgentResult:
        # Fresh conversation for this run: system prompt + the user's message.
        self.memory.clear()
        self.memory.add({"role": "system", "content": self.system_prompt})
        self.memory.add({"role": "user", "content": user_message})

        all_tool_calls = []
        usage = {}

        # AGENT_START — a hook can vet the input and abort before we begin.
        ev = self.hooks.emit(HookEvent(type=EventType.AGENT_START, data={"user_message": user_message}))
        if ev.data.get("abort"):
            return self._aborted("aborted at start: " + str(ev.data.get("abort_reason", "")),
                                 0, all_tool_calls, usage)

        max_turns = self.max_tool_calls + 1   # tool-call turns + one answer turn
        last_text = ""

        for turn in range(1, max_turns + 1):
            # LLM_CALL — a hook can modify the messages or abort the call.
            messages = self.memory.get_messages()
            ev = self.hooks.emit(HookEvent(type=EventType.LLM_CALL, turn=turn, messages=messages))
            if ev.data.get("abort"):
                return self._aborted("aborted: " + str(ev.data.get("abort_reason", "")),
                                     turn, all_tool_calls, usage)
            messages = ev.messages if ev.messages is not None else messages

            # Tools are available every turn; force a tool on turn 1 only.
            tool_choice = "required" if (turn == 1 and self.tools) else "auto"
            resp = self.llm.chat(
                messages,
                tools=[t.to_openai_schema() for t in self.tools] if self.tools else None,
                tool_choice=tool_choice,
            )
            msg = resp.choices[0].message

            # LLM_RESPONSE — a hook can inspect/modify the response or abort.
            ev = self.hooks.emit(HookEvent(type=EventType.LLM_RESPONSE, turn=turn,
                                           messages=messages, response=msg))
            if ev.data.get("abort"):
                return self._aborted("aborted: " + str(ev.data.get("abort_reason", "")),
                                     turn, all_tool_calls, usage)
            msg = ev.response if ev.response is not None else msg

            self.memory.add(msg.model_dump())
            last_text = msg.content or ""
            if getattr(resp, "usage", None):
                usage = self._usage_dict(resp.usage)

            # No tool call -> this is the final answer.
            if not msg.tool_calls:
                ev = HookEvent(type=EventType.AGENT_END, turn=turn,
                               messages=self.memory.get_messages())
                ev.data["final_text"] = last_text   # make it available to hooks
                ev = self.hooks.emit(ev)
                final_text = ev.data.get("final_text", last_text) if ev.data else last_text
                handoff = ev.data.get("handoff")   # a hook can set this to hand off
                return AgentResult(text=final_text, turns=turn, tool_calls=all_tool_calls,
                                   usage=usage, handoff=handoff)

            # Tool call(s): run each, feed results back, and loop.
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                all_tool_calls.append({"name": name, "args": args})
                result = self._run_tool(tc, turn, args)
                self.memory.add({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Ran out of turns without a final answer -> graceful abort.
        return self._aborted(f"max tool calls ({self.max_tool_calls}) reached",
                             max_turns, all_tool_calls, usage, text=last_text)

    # ------------------------------------------------------------- helpers
    def _run_tool(self, tc, turn, args) -> str:
        name = tc.function.name

        ev = self.hooks.emit(HookEvent(type=EventType.TOOL_CALL, turn=turn,
                                       tool_name=name, tool_args=args))
        if ev.data.get("abort"):
            return f"[aborted: {ev.data.get('abort_reason', 'tool call blocked')}]"
        args = ev.tool_args if ev.tool_args is not None else args

        tool_obj = self._find_tool(name)
        if tool_obj is None:
            return f"[error: unknown tool '{name}']"

        try:
            result = tool_obj(args)   # __call__ handles sync/async execution
        except Exception as e:   # a crashing tool must not kill the agent
            result = f"[error: {type(e).__name__}: {e}]"

        if isinstance(result, (dict, list)):
            result = json.dumps(result)

        ev = self.hooks.emit(HookEvent(type=EventType.TOOL_RESULT, turn=turn,
                                       tool_name=name, tool_args=args, tool_result=result))
        return ev.tool_result if ev.tool_result is not None else result

    def _find_tool(self, name: str):
        for t in self.tools:
            if t.name == name:
                return t
        return None

    @staticmethod
    def _aborted(reason, turns, tool_calls, usage, text="") -> AgentResult:
        return AgentResult(text=text, turns=turns, tool_calls=tool_calls,
                           usage=usage, aborted=True, abort_reason=reason)

    @staticmethod
    def _usage_dict(usage) -> dict:
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
