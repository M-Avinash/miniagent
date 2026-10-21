"""Offline unit tests for the miniagent framework.

These run with no network and no LLM — safe for CI. Run:

    pytest miniagent/tests/test_unit.py
    # or the whole offline suite:
    pytest miniagent/tests/ -m "not live"
"""

import io
import json
import os
import tempfile
from contextlib import redirect_stderr

import pytest

from miniagent import (
    Agent, LLMClient, Memory, Tool, tool,
    HookEvent, HookFn, HookPipeline, EventType, AgentResult,
    build_structured_prompt, extract_json, parse_structured,
    run_python, verbose_log, trace_hook, ObservationalMemory,
    HandoffOrchestrator, make_handoff_tool, handoff_hook,
    make_supervisor_tool, make_agent_tool,
    GraphMemory, graph_memory_hook, graph_injection_hook,
)


# ─── 1. Events ───────────────────────────────────────────────────────────
class TestEvents:
    def test_event_types(self):
        assert EventType.AGENT_START == "agent_start"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.AGENT_END == "agent_end"

    def test_hook_event_defaults(self):
        ev = HookEvent(type=EventType.AGENT_START, data={"user_message": "hi"})
        assert ev.turn == 0
        assert ev.data == {"user_message": "hi"}

    def test_agent_result_defaults(self):
        r = AgentResult(text="hello", turns=1)
        assert r.aborted is False
        assert r.tool_calls == []


# ─── 2. Memory ───────────────────────────────────────────────────────────
class TestMemory:
    def test_trims_to_max(self):
        m = Memory(max_messages=5)
        m.add({"role": "system", "content": "You are helpful."})
        for i in range(10):
            m.add({"role": "user", "content": f"msg {i}"})
        msgs = m.get_messages()
        assert len(msgs) <= 5
        assert msgs[0]["role"] == "system"

    def test_clear(self):
        m = Memory(max_messages=5)
        m.add({"role": "user", "content": "x"})
        m.clear()
        assert len(m.get_messages()) == 0

    def test_system_always_first(self):
        m = Memory(max_messages=3)
        m.add({"role": "system", "content": "sys"})
        m.add({"role": "user", "content": "1"})
        m.add({"role": "assistant", "content": "2"})
        m.add({"role": "user", "content": "3"})
        m.add({"role": "assistant", "content": "4"})
        msgs = m.get_messages()
        assert msgs[0]["role"] == "system"
        assert len(msgs) <= 3

    def test_trim_never_orphans_tool_message(self):
        """Trimming must drop an assistant tool_calls message together with
        the tool messages that answer it, so no `tool` message is left
        without its matching tool_call_id."""
        m = Memory(max_messages=4)
        m.add({"role": "system", "content": "sys"})
        m.add({"role": "user", "content": "q"})
        m.add({"role": "assistant", "content": None,
               "tool_calls": [{"id": "call_1", "type": "function",
                               "function": {"name": "t", "arguments": "{}"}}]})
        m.add({"role": "tool", "tool_call_id": "call_1", "content": "result"})
        m.add({"role": "assistant", "content": "final"})
        msgs = m.get_messages()
        # Every tool message must have a matching assistant tool_call present.
        call_ids = {tc["id"] for msg in msgs
                    if msg.get("role") == "assistant" and msg.get("tool_calls")
                    for tc in msg["tool_calls"]}
        for msg in msgs:
            if msg.get("role") == "tool":
                assert msg["tool_call_id"] in call_ids, "orphaned tool message"


# ─── 3. Tool decorator & schema ──────────────────────────────────────────
@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers.
    Args:
        a (int): First number.
        b (int): Second number.
    """
    return str(a + b)


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone.
    Args:
        name (str): Name to greet.
        excited (bool): Whether to use exclamation.
    """
    return f"Hello, {name}{'!' if excited else '.'}"


@tool(name="custom_name")
def my_func(x: str) -> str:
    """Test."""
    return x


@tool
def return_dict(x: str) -> dict:
    """Return a dict.
    Args:
        x (str): Input.
    """
    return {"key": x}


class TestTool:
    def test_name(self):
        assert add_numbers.name == "add_numbers"

    def test_schema(self):
        schema = add_numbers.to_openai_schema()
        assert schema["type"] == "function"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "a" in schema["function"]["parameters"]["required"]

    def test_execute(self):
        assert add_numbers({"a": 3, "b": 4}) == "7"
        assert add_numbers(a=10, b=20) == "30"

    def test_optional_param(self):
        assert "excited" not in greet.to_openai_schema()["function"]["parameters"]["required"]
        assert greet({"name": "World"}) == "Hello, World."

    def test_name_override(self):
        assert my_func.name == "custom_name"

    def test_dict_result_is_json(self):
        r = return_dict({"x": "test"})
        assert isinstance(r, str) and "key" in r


# ─── 4. Structured output ────────────────────────────────────────────────
class TestStructured:
    def test_extract_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_extract_fenced(self):
        assert extract_json('```json\n{"b": 2}\n```') == {"b": 2}

    def test_extract_embedded(self):
        assert extract_json('The answer is {"c": 3} ok?') == {"c": 3}

    def test_extract_array(self):
        assert extract_json('result: [1,2,3]') == [1, 2, 3]

    def test_extract_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here")

    def test_parse_structured_fields(self):
        assert parse_structured('{"x": 1}', ["x"]) == {"x": 1}

    def test_parse_structured_missing_field(self):
        with pytest.raises(ValueError):
            parse_structured('{"x": 1}', ["y"])

    def test_build_structured_prompt(self):
        prompt = build_structured_prompt({"type": "object"}, "Be brief.")
        assert "JSON" in prompt


# ─── 5. Sandbox ──────────────────────────────────────────────────────────
class TestSandbox:
    def test_basic(self):
        r = run_python({"code": "print(2+2)"})
        assert "[exit 0]" in r and "4" in r

    def test_compute(self):
        r = run_python({"code": "x = [i**2 for i in range(5)]; print(sum(x))"})
        assert "30" in r

    def test_error(self):
        r = run_python({"code": "import nonexist_module_xyz"})
        assert "[exit 1]" in r or "ModuleNotFoundError" in r

    def test_timeout(self):
        r = run_python({"code": "import time; time.sleep(10)", "timeout": 1})
        assert "[timeout]" in r

    def test_stdout(self):
        r = run_python({"code": "print('hello')"})
        assert "hello" in r


# ─── 6. Hooks ────────────────────────────────────────────────────────────
class TestHooks:
    def test_hook_fires(self):
        seen = []
        pipeline = HookPipeline([lambda e: (seen.append(e.type), e)[1]])
        pipeline.emit(HookEvent(type=EventType.AGENT_START))
        assert EventType.AGENT_START in seen

    def test_buggy_hook_skipped(self):
        seen = []
        def buggy(event):
            raise RuntimeError("boom")
        pipeline = HookPipeline([buggy, lambda e: (seen.append(e.type), e)[1]])
        pipeline.emit(HookEvent(type=EventType.LLM_CALL, turn=1, messages=[]))
        assert EventType.LLM_CALL in seen

    def test_hook_abort(self):
        def abort_hook(event):
            if event.type == EventType.AGENT_START:
                event.data["abort"] = True
                event.data["abort_reason"] = "test abort"
            return event
        ev = HookPipeline([abort_hook]).emit(HookEvent(type=EventType.AGENT_START, data={}))
        assert ev.data.get("abort") is True

    def test_hook_mutation(self):
        def mutate(event):
            event.data["injected"] = True
            return event
        ev = HookPipeline([mutate]).emit(HookEvent(type=EventType.TOOL_CALL, data={}))
        assert ev.data.get("injected") is True


# ─── 7. verbose_log ──────────────────────────────────────────────────────
class TestVerboseLog:
    def test_prints(self):
        log_hook = verbose_log("test-agent")
        buf = io.StringIO()
        with redirect_stderr(buf):
            log_hook(HookEvent(type=EventType.AGENT_START, data={"user_message": "test msg"}))
        output = buf.getvalue()
        assert "[test-agent]" in output and "START" in output


# ─── 8. ObservationalMemory ──────────────────────────────────────────────
class TestObservationalMemory:
    def test_store_and_recall(self, tmp_path):
        p = str(tmp_path / "obs.jsonl")
        om = ObservationalMemory("test-agent", path=p, max_entries=10)
        om.hook(HookEvent(type=EventType.AGENT_END, turn=3, messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]))
        assert len(om.recall()) == 1
        assert om.recall()[0]["user"] == "What is 2+2?"
        assert om.recall()[0]["assistant"] == "4"
        assert "What is 2+2?" in om.recall_prompt()

    def test_persistence(self, tmp_path):
        p = str(tmp_path / "obs.jsonl")
        om = ObservationalMemory("test-agent", path=p)
        om.hook(HookEvent(type=EventType.AGENT_END, turn=1, messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
        ]))
        om2 = ObservationalMemory("test-agent", path=p)
        assert len(om2.recall()) == 1


# ─── 9. GraphMemory ──────────────────────────────────────────────────────
class TestGraphMemory:
    def test_empty(self):
        g = GraphMemory()
        assert g.focus_path() == "(empty)"
        assert g.prompt_block() == ""

    def test_add_and_query(self):
        g = GraphMemory()
        g.add_record("Customer", {"CustomerID": "ALFKI", "Name": "Alfred"}, "CustomerID")
        g.add_record("Order", {"OrderID": 10248, "CustomerID": "ALFKI"}, "OrderID")
        assert "Customer" in g.focus_path() and "Order" in g.focus_path()
        assert "ALFKI" in g.active_keys().get("Customer", [])
        assert g.get_record("Customer", "ALFKI")["Name"] == "Alfred"
        assert "SEMANTIC GRAPH MEMORY" in g.prompt_block()
        g.clear()
        assert g.focus_path() == "(empty)"

    def test_memory_hook_populates(self):
        g = GraphMemory()
        hook = graph_memory_hook(g, {"Order": "OrderID"})
        result_json = json.dumps({"value": [{"OrderID": 1, "CustomerID": "A"},
                                            {"OrderID": 2, "CustomerID": "B"}]})
        hook(HookEvent(type=EventType.TOOL_RESULT, tool_result=result_json))
        assert len(g.active_keys().get("Order", [])) == 2

    def test_injection_hook_modifies(self):
        g = GraphMemory()
        g.add_record("Order", {"OrderID": 99}, "OrderID")
        hook = graph_injection_hook(g)
        msgs = [{"role": "system", "content": "You are helpful."}]
        ev = hook(HookEvent(type=EventType.LLM_CALL, messages=msgs))
        assert "SEMANTIC GRAPH" in ev.messages[0]["content"]


# ─── 10. Handoff tool & hook ─────────────────────────────────────────────
class TestHandoff:
    def test_tool(self):
        ht = make_handoff_tool(["agent_b"])
        assert ht.name == "handoff"
        assert "[handoff: agent_b]" in ht({"target_agent": "agent_b"})

    def test_hook(self):
        ev = HookEvent(type=EventType.TOOL_CALL, tool_name="handoff",
                       tool_args={"target_agent": "agent_b"}, data={})
        handoff_hook(ev)
        assert ev.data.get("handoff") == "agent_b"
        assert ev.data.get("abort") is True


# ─── 11. LLMClient config ────────────────────────────────────────────────
class TestLLMClientConfig:
    def test_config(self):
        llm = LLMClient()
        assert llm.model is not None
        assert hasattr(llm, "chat")
        assert llm.max_retries >= 0


# ─── 14. HandoffOrchestrator shared_state & session (no LLM) ─────────────
class TestOrchestratorState:
    def _agents(self):
        @tool
        def compute(x: int) -> str:
            """Compute double of x.
            Args:
                x (int): The number.
            """
            return str(x * 2)
        a = Agent("Hand off.", tools=[make_handoff_tool(["b"])],
                  hooks=[handoff_hook], name="a", max_tool_calls=3)
        b = Agent("Answer.", tools=[compute], name="b", max_tool_calls=3)
        return a, b

    def test_shared_state_created(self):
        a, b = self._agents()
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b})
        assert orch.shared_state is not None
        assert "session" in orch.shared_state
        assert orch.shared_state["session"] == []

    def test_passed_shared_state(self):
        a, b = self._agents()
        my_state = {"entities": {"Orders": []}, "custom": "value"}
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b},
                                   shared_state=my_state)
        assert orch.shared_state is my_state
        assert orch.shared_state["custom"] == "value"
        assert "session" in my_state

    def test_build_user_message_no_session(self):
        a, b = self._agents()
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b})
        msg = orch._build_user_message("What is 2+2?")
        assert "Original question: What is 2+2?" in msg
        assert "Previous agent result:" not in msg

    def test_build_user_message_with_session(self):
        a, b = self._agents()
        my_state = {"session": [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]}
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b},
                                   shared_state=my_state)
        msg = orch._build_user_message("What is 4+4?")
        assert "Original question: What is 4+4?" in msg
        assert "Previous Q&A in this session:" in msg
        assert "What is 2+2?" in msg

    def test_build_user_message_with_previous_result(self):
        a, b = self._agents()
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b})
        msg = orch._build_user_message("What is 4+4?", previous_result="[handoff: worker]")
        assert "Previous agent result: [handoff: worker]" in msg
        assert "Original question: What is 4+4?" in msg
        assert msg.count("Original question:") == 1

    def test_reset_session(self):
        a, b = self._agents()
        my_state = {"session": [{"question": "q", "answer": "a"}]}
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b},
                                   shared_state=my_state)
        orch.reset_session()
        assert my_state["session"] == []


# ─── 16. Northwind example imports (no network) ──────────────────────────
class TestNorthwindImports:
    def test_tools_are_tools(self):
        from miniagent.examples.northwind.tools import odata_query, analyze_data
        assert isinstance(odata_query, Tool)
        assert isinstance(analyze_data, Tool)

    def test_field_descriptions(self):
        from miniagent.examples.northwind.schema import FIELD_DESCRIPTIONS
        for entity in ["Orders", "Products", "Customers", "Employees",
                       "Categories", "Order_Details"]:
            assert len(FIELD_DESCRIPTIONS[entity]) > 0

    def test_agents_construct(self):
        from miniagent.examples.northwind.agents import (
            create_orchestrator_agent, create_data_fetcher_agent,
            create_data_analyst_agent,
        )
        # Construction must not require network.
        assert create_orchestrator_agent() is not None
        assert create_data_fetcher_agent() is not None
        assert create_data_analyst_agent() is not None


# ─── 17. Multi-entity shared state ───────────────────────────────────────
class TestMultiEntitySharedState:
    def test_entities(self):
        from miniagent.examples.northwind.tools import shared_state
        shared_state["entities"].clear()
        shared_state["entities"]["Orders"] = {"value": [{"OrderID": 1, "ShipCountry": "Germany"}]}
        shared_state["entities"]["Products"] = {"value": [{"ProductID": 1, "ProductName": "Chai"}]}
        assert len(shared_state["entities"]) == 2
        assert shared_state["entities"]["Orders"]["value"][0]["ShipCountry"] == "Germany"
        shared_state["data_desc"] = ", ".join(
            f"{k}: {len(v.get('value', []))} records" for k, v in shared_state["entities"].items()
        )
        assert "Orders" in shared_state["data_desc"]
        assert "Products" in shared_state["data_desc"]
        shared_state["entities"].clear()
        assert len(shared_state["entities"]) == 0


# ─── 18. make_supervisor_tool (offline: tool shape + error paths) ────────
class TestSupervisorTool:
    def test_tool_name_and_schema(self):
        a = Agent("worker a", name="a")
        b = Agent("worker b", name="b")
        t = make_supervisor_tool({"a": a, "b": b})
        assert t.name == "call_worker"
        schema = t.to_openai_schema()
        assert "agent_name" in schema["function"]["parameters"]["properties"]
        assert "task" in schema["function"]["parameters"]["properties"]

    def test_unknown_worker_returns_error(self):
        a = Agent("worker a", name="a")
        t = make_supervisor_tool({"a": a})
        result = t({"agent_name": "nonexistent", "task": "do something"})
        assert "[error: unknown worker" in result

    def test_worker_abort_returns_error(self):
        """If the worker aborts, the supervisor gets an error string."""
        def abort_hook(event):
            if event.type == EventType.AGENT_START:
                event.data["abort"] = True
                event.data["abort_reason"] = "blocked"
            return event
        a = Agent("worker", hooks=[abort_hook], name="a")
        t = make_supervisor_tool({"a": a})
        result = t({"agent_name": "a", "task": "do something"})
        assert "[error: a aborted" in result


# ─── 19. make_agent_tool (offline: tool shape + error paths) ─────────────
class TestAgentTool:
    def test_tool_name(self):
        a = Agent("researcher", name="researcher")
        t = make_agent_tool(a)
        assert t.name == "call_researcher"

    def test_description_override(self):
        a = Agent("researcher", name="researcher")
        t = make_agent_tool(a, description="Custom description")
        assert t.to_openai_schema()["function"]["description"] == "Custom description"

    def test_unknown_agent_name_fallback(self):
        a = Agent("no name")
        t = make_agent_tool(a)
        assert t.name == "call_sub_agent"


# ─── 20. trace_hook (offline: writes JSONL) ──────────────────────────────
class TestTraceHook:
    def test_writes_jsonl(self, tmp_path):
        p = str(tmp_path / "trace.jsonl")
        hook = trace_hook(p, run_id="test-run")
        hook(HookEvent(type=EventType.AGENT_START, data={"user_message": "hi"}))
        hook(HookEvent(type=EventType.TOOL_CALL, turn=1, tool_name="test", tool_args={"x": 1}))
        hook(HookEvent(type=EventType.AGENT_END, turn=1))
        lines = open(p).readlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["run_id"] == "test-run"
        assert first["event"] == "agent_start"
        second = json.loads(lines[1])
        assert second["tool_name"] == "test"
        assert second["tool_args"] == {"x": 1}

    def test_auto_run_id(self, tmp_path):
        p = str(tmp_path / "trace.jsonl")
        hook = trace_hook(p)
        hook(HookEvent(type=EventType.AGENT_START, data={}))
        line = json.loads(open(p).readline())
        assert "run_id" in line and len(line["run_id"]) > 0


# ─── 21. Tool result caching (offline) ───────────────────────────────────
class TestToolCaching:
    def test_cache_key_computation(self):
        """Verify the cache key is deterministic for the same args."""
        import json as _json
        args1 = {"b": 2, "a": 1}
        args2 = {"a": 1, "b": 2}
        k1 = ("tool", _json.dumps(args1, sort_keys=True, default=str))
        k2 = ("tool", _json.dumps(args2, sort_keys=True, default=str))
        assert k1 == k2

    def test_different_args_different_keys(self):
        import json as _json
        k1 = ("tool", _json.dumps({"a": 1}, sort_keys=True, default=str))
        k2 = ("tool", _json.dumps({"a": 2}, sort_keys=True, default=str))
        assert k1 != k2


# ─── 22. on_failure in orchestrator (offline: callback wiring) ──────────
class TestOnFailure:
    def test_callback_stored(self):
        a = Agent("a", name="a")
        b = Agent("b", name="b")
        def my_callback(agent, result, question):
            return None
        orch = HandoffOrchestrator([a, b], agent_names={"a": a, "b": b},
                                   on_failure=my_callback)
        assert orch.on_failure is my_callback

    def test_default_none(self):
        a = Agent("a", name="a")
        orch = HandoffOrchestrator([a])
        assert orch.on_failure is None


# ─── 23. _build_output_schema (offline: no eval) ─────────────────────────
class TestBuildOutputSchema:
    def test_dataclass(self):
        from dataclasses import dataclass
        @dataclass
        class Answer:
            city: str
            population: int
        schema = Agent._build_output_schema(Answer)
        assert schema["type"] == "object"
        assert schema["properties"]["city"]["type"] == "string"
        assert schema["properties"]["population"]["type"] == "integer"
        assert "city" in schema["required"]
        assert "population" in schema["required"]

    def test_dataclass_with_default(self):
        from dataclasses import dataclass, field
        @dataclass
        class Config:
            name: str
            verbose: bool = False
        schema = Agent._build_output_schema(Config)
        assert "name" in schema["required"]
        assert "verbose" not in schema["required"]

    def test_forward_reference(self):
        """Forward references (string annotations) must resolve without eval."""
        from dataclasses import dataclass
        @dataclass
        class Item:
            name: str
            tags: "list[str]"
        schema = Agent._build_output_schema(Item)
        assert schema["properties"]["tags"]["type"] == "array"

    def test_unknown_type_defaults_to_string(self):
        from dataclasses import dataclass
        @dataclass
        class Weird:
            x: "SomeUnknownType"
        schema = Agent._build_output_schema(Weird)
        assert schema["properties"]["x"]["type"] == "string"
