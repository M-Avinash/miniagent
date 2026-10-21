"""Comprehensive tests for the miniagent framework.

Run: python miniagent/tests/test_framework.py
"""

import json
import os
import sys
import tempfile

# Ensure we can import miniagent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from miniagent import (
    Agent, LLMClient, Memory, Tool, tool,
    HookEvent, HookFn, HookPipeline, EventType, AgentResult,
    build_structured_prompt, extract_json, parse_structured,
    run_python, verbose_log, ObservationalMemory,
    HandoffOrchestrator, make_handoff_tool, handoff_hook,
    GraphMemory, graph_memory_hook, graph_injection_hook,
)

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  \u2717 {name} \u2014 {detail}")


print("=" * 60)
print("miniagent framework tests")
print("=" * 60)

# ─── 1. Events ───────────────────────────────────────────────────────────
print("\n[1] Events")
check("EventType has AGENT_START", EventType.AGENT_START == "agent_start")
check("EventType has TOOL_CALL", EventType.TOOL_CALL == "tool_call")
check("EventType has AGENT_END", EventType.AGENT_END == "agent_end")
ev = HookEvent(type=EventType.AGENT_START, data={"user_message": "hi"})
check("HookEvent default turn=0", ev.turn == 0)
check("HookEvent data dict", ev.data == {"user_message": "hi"})
r = AgentResult(text="hello", turns=1)
check("AgentResult defaults", r.aborted == False and r.tool_calls == [])

# ─── 2. Memory ───────────────────────────────────────────────────────────
print("\n[2] Memory")
m = Memory(max_messages=5)
m.add({"role": "system", "content": "You are helpful."})
for i in range(10):
    m.add({"role": "user", "content": f"msg {i}"})
msgs = m.get_messages()
check("Memory trims to max", len(msgs) <= 5, f"got {len(msgs)}")
check("Memory keeps system prompt", msgs[0]["role"] == "system")
m.clear()
check("Memory clear", len(m.get_messages()) == 0)

# ─── 3. Tool decorator & schema ──────────────────────────────────────────
print("\n[3] Tool decorator & schema")

@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers.
    Args:
        a (int): First number.
        b (int): Second number.
    """
    return str(a + b)

check("Tool name", add_numbers.name == "add_numbers")
schema = add_numbers.to_openai_schema()
check("Schema type=function", schema["type"] == "function")
check("Schema has params", "a" in schema["function"]["parameters"]["properties"])
check("Schema required", "a" in schema["function"]["parameters"]["required"])
check("Tool executes", add_numbers({"a": 3, "b": 4}) == "7")
check("Tool kwargs", add_numbers(a=10, b=20) == "30")

# Optional params
@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone.
    Args:
        name (str): Name to greet.
        excited (bool): Whether to use exclamation.
    """
    return f"Hello, {name}{'!' if excited else '.'}"

check("Optional param not required", "excited" not in greet.to_openai_schema()["function"]["parameters"]["required"])
check("Default param works", greet({"name": "World"}) == "Hello, World.")

# @tool with name override
@tool(name="custom_name")
def my_func(x: str) -> str:
    """Test."""
    return x
check("Tool name override", my_func.name == "custom_name")

# ─── 4. Structured output ────────────────────────────────────────────────
print("\n[4] Structured output")
check("extract_json plain", extract_json('{"a": 1}') == {"a": 1})
check("extract_json fenced", extract_json('```json\n{"b": 2}\n```') == {"b": 2})
check("extract_json embedded", extract_json('The answer is {"c": 3} ok?') == {"c": 3})
check("extract_json array", extract_json('result: [1,2,3]') == [1,2,3])
try:
    extract_json("no json here")
    check("extract_json raises", False, "should have raised")
except ValueError:
    check("extract_json raises", True)
check("parse_structured fields", parse_structured('{"x": 1}', ["x"]) == {"x": 1})
try:
    parse_structured('{"x": 1}', ["y"])
    check("parse_structured missing field", False, "should have raised")
except ValueError:
    check("parse_structured missing field", True)
prompt = build_structured_prompt({"type": "object"}, "Be brief.")
check("build_structured_prompt has JSON", "JSON" in prompt)

# ─── 5. Sandbox ──────────────────────────────────────────────────────────
print("\n[5] Sandbox (run_python)")
r = run_python({"code": "print(2+2)"})
check("run_python basic", "[exit 0]" in r and "4" in r, r)
r = run_python({"code": "x = [i**2 for i in range(5)]; print(sum(x))"})
check("run_python compute", "30" in r, r)
r = run_python({"code": "import nonexist_module_xyz"})
check("run_python error", "[exit 1]" in r or "ModuleNotFoundError" in r, r)
r = run_python({"code": "import time; time.sleep(10)", "timeout": 1})
check("run_python timeout", "[timeout]" in r, r)
r = run_python({"code": "print('hello')"})
check("run_python stdout", "hello" in r, r)

# ─── 6. Hooks ────────────────────────────────────────────────────────────
print("\n[6] Hooks")
events_seen = []
def tracking_hook(event):
    events_seen.append(event.type)
    return event

pipeline = HookPipeline([tracking_hook])
ev = pipeline.emit(HookEvent(type=EventType.AGENT_START))
check("Hook fires", EventType.AGENT_START in events_seen)

# Buggy hook doesn't crash
def buggy_hook(event):
    raise RuntimeError("boom")
pipeline2 = HookPipeline([buggy_hook, tracking_hook])
events_seen.clear()
ev = pipeline2.emit(HookEvent(type=EventType.LLM_CALL, turn=1, messages=[]))
check("Buggy hook skipped", EventType.LLM_CALL in events_seen)

# Hook can abort
def abort_hook(event):
    if event.type == EventType.AGENT_START:
        event.data["abort"] = True
        event.data["abort_reason"] = "test abort"
    return event
pipeline3 = HookPipeline([abort_hook])
ev = pipeline3.emit(HookEvent(type=EventType.AGENT_START, data={}))
check("Hook abort flag", ev.data.get("abort") == True)

# Hook can mutate
def mutate_hook(event):
    event.data["injected"] = True
    return event
ev = HookPipeline([mutate_hook]).emit(HookEvent(type=EventType.TOOL_CALL, data={}))
check("Hook mutation", ev.data.get("injected") == True)

# ─── 7. verbose_log ──────────────────────────────────────────────────────
print("\n[7] verbose_log")
import io
from contextlib import redirect_stderr
log_hook = verbose_log("test-agent")
buf = io.StringIO()
with redirect_stderr(buf):
    log_hook(HookEvent(type=EventType.AGENT_START, data={"user_message": "test msg"}))
output = buf.getvalue()
check("verbose_log prints", "[test-agent]" in output and "START" in output, output)

# ─── 8. ObservationalMemory ──────────────────────────────────────────────
print("\n[8] ObservationalMemory")
tmp_path = tempfile.mktemp(suffix=".jsonl")
om = ObservationalMemory("test-agent", path=tmp_path, max_entries=10)
ev = HookEvent(type=EventType.AGENT_END, turn=3, messages=[
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
])
om.hook(ev)
check("Observation stored", len(om.recall()) == 1)
check("Observation has user", om.recall()[0]["user"] == "What is 2+2?")
check("Observation has assistant", om.recall()[0]["assistant"] == "4")
prompt = om.recall_prompt()
check("recall_prompt has content", "What is 2+2?" in prompt)
om2 = ObservationalMemory("test-agent", path=tmp_path)
check("Persistence reload", len(om2.recall()) == 1)
os.unlink(tmp_path)

# ─── 9. GraphMemory ──────────────────────────────────────────────────────
print("\n[9] GraphMemory")
g = GraphMemory()
check("Empty path", g.focus_path() == "(empty)")
check("Empty prompt block", g.prompt_block() == "")
g.add_record("Customer", {"CustomerID": "ALFKI", "Name": "Alfred"}, "CustomerID")
g.add_record("Order", {"OrderID": 10248, "CustomerID": "ALFKI"}, "OrderID")
check("Path built", "Customer" in g.focus_path() and "Order" in g.focus_path())
check("Active keys", "ALFKI" in g.active_keys().get("Customer", []))
check("Get record", g.get_record("Customer", "ALFKI")["Name"] == "Alfred")
check("Prompt block", "SEMANTIC GRAPH MEMORY" in g.prompt_block())
g.clear()
check("Clear works", g.focus_path() == "(empty)")

g2 = GraphMemory()
hook = graph_memory_hook(g2, {"Order": "OrderID"})
result_json = json.dumps({"value": [{"OrderID": 1, "CustomerID": "A"}, {"OrderID": 2, "CustomerID": "B"}]})
ev = HookEvent(type=EventType.TOOL_RESULT, tool_result=result_json)
hook(ev)
check("graph_memory_hook populates", len(g2.active_keys().get("Order", [])) == 2)

g3 = GraphMemory()
g3.add_record("Order", {"OrderID": 99}, "OrderID")
injection_hook = graph_injection_hook(g3)
msgs = [{"role": "system", "content": "You are helpful."}]
ev = HookEvent(type=EventType.LLM_CALL, messages=msgs)
injection_hook(ev)
check("graph_injection_hook modifies", "SEMANTIC GRAPH" in ev.messages[0]["content"])

# ─── 10. Handoff tool & hook ─────────────────────────────────────────────
print("\n[10] Handoff tool & hook")
ht = make_handoff_tool(["agent_b"])
check("Handoff tool name", ht.name == "handoff")
check("Handoff tool executes", "[handoff: agent_b]" in ht({"target_agent": "agent_b"}))

ev = HookEvent(type=EventType.TOOL_CALL, tool_name="handoff", tool_args={"target_agent": "agent_b"}, data={})
handoff_hook(ev)
check("handoff_hook sets handoff", ev.data.get("handoff") == "agent_b")
check("handoff_hook sets abort", ev.data.get("abort") == True)

# ─── 11. LLMClient config ────────────────────────────────────────────────
print("\n[11] LLMClient config")
llm = LLMClient()
check("LLMClient model set", llm.model is not None)
check("LLMClient has chat method", hasattr(llm, "chat"))
check("LLMClient retry count", llm.max_retries >= 0)

# ─── 12. Agent (integration with LLM) ────────────────────────────────────
print("\n[12] Agent integration (live LLM calls)")

call_count = {"n": 0}
@tool
def get_answer(question: str) -> str:
    """Get the answer to a question.
    Args:
        question (str): The question to answer.
    """
    call_count["n"] += 1
    return "42"

agent = Agent(
    "You are a helpful assistant. Use get_answer for any question.",
    tools=[get_answer],
    max_tool_calls=3,
    name="test",
)
result = agent.run("What is the answer to life?")
check("Agent produces text", len(result.text) > 0, f"got: {result.text[:50]}")
check("Agent used tool", call_count["n"] >= 1)
check("Agent result has turns", result.turns >= 1)
check("Agent not aborted", result.aborted == False)

agent2 = Agent("You are a helpful assistant.")
result2 = agent2.run("Say 'hello world' exactly.")
check("No-tool agent works", "hello" in result2.text.lower(), result2.text[:50])

def immediate_abort(event):
    if event.type == EventType.AGENT_START:
        event.data["abort"] = True
        event.data["abort_reason"] = "blocked"
    return event

agent3 = Agent("Test", hooks=[immediate_abort])
result3 = agent3.run("This should abort")
check("Abort hook works", result3.aborted == True)
check("Abort reason", "blocked" in (result3.abort_reason or ""))

call_n = {"n": 0}
@tool
def always_call(x: str) -> str:
    """Always returns a result that encourages more calls.
    Args:
        x (str): Input.
    """
    call_n["n"] += 1
    return "Please call me again with a different value."

agent4 = Agent(
    "You must call always_call 5 times before answering.",
    tools=[always_call],
    max_tool_calls=3,
)
result4 = agent4.run("Start the process")
check("Max tool calls enforced", result4.aborted == True or result4.turns <= 4)

# ─── 13. HandoffOrchestrator (integration) ───────────────────────────────
print("\n[13] HandoffOrchestrator (integration)")

@tool
def compute(x: int) -> str:
    """Compute double of x.
    Args:
        x (int): The number.
    """
    return str(x * 2)

agent_a = Agent(
    "You receive a number. Hand off to 'agent_b' with the number.",
    tools=[make_handoff_tool(["agent_b"])],
    hooks=[handoff_hook],
    name="agent_a",
    max_tool_calls=3,
)
agent_b = Agent(
    "You receive context with a number. Use compute to double it and give the final answer.",
    tools=[compute],
    name="agent_b",
    max_tool_calls=3,
)

orch = HandoffOrchestrator(
    [agent_a, agent_b],
    max_handoffs=3,
    agent_names={"agent_a": agent_a, "agent_b": agent_b},
)
result5 = orch.run("Double the number 21")
check("Orchestrator produces answer", len(result5.text) > 0, result5.text[:80])
check("Orchestrator not aborted", result5.aborted == False)
check("Answer contains 42", "42" in result5.text, result5.text[:80])

# ─── 14. HandoffOrchestrator shared_state & session ──────────────────────
print("\n[14] HandoffOrchestrator shared_state & session")

# shared_state is created automatically
orch_ss = HandoffOrchestrator([agent_a, agent_b], agent_names={"agent_a": agent_a, "agent_b": agent_b})
check("shared_state created", orch_ss.shared_state is not None)
check("session key exists", "session" in orch_ss.shared_state)
check("session starts empty", orch_ss.shared_state["session"] == [])

# shared_state can be passed in
my_state = {"entities": {"Orders": []}, "custom": "value"}
orch_ss2 = HandoffOrchestrator([agent_a, agent_b], agent_names={"agent_a": agent_a, "agent_b": agent_b},
                                shared_state=my_state)
check("Passed shared_state used", orch_ss2.shared_state is my_state)
check("Custom key preserved", orch_ss2.shared_state["custom"] == "value")
check("Session added to passed state", "session" in my_state)

# _build_user_message with no session, no previous result
msg = orch_ss2._build_user_message("What is 2+2?")
check("No session: has original question", "Original question: What is 2+2?" in msg)
check("No session: no previous result", "Previous agent result:" not in msg)

# _build_user_message with session
my_state["session"] = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3+3?", "answer": "6"},
]
msg2 = orch_ss2._build_user_message("What is 4+4?")
check("Session: has original question", "Original question: What is 4+4?" in msg2)
check("Session: has 'Previous Q&A'", "Previous Q&A in this session:" in msg2)
check("Session: has Q1", "What is 2+2?" in msg2)
check("Session: has A1", "4" in msg2)

# _build_user_message with previous result
msg3 = orch_ss2._build_user_message("What is 4+4?", previous_result="[handoff: worker]")
check("Handoff: has previous result", "Previous agent result: [handoff: worker]" in msg3)
check("Handoff: has original question", "Original question: What is 4+4?" in msg3)
check("Handoff: flat structure (no nesting)", msg3.count("Original question:") == 1)

# reset_session
orch_ss2.reset_session()
check("reset_session clears", my_state["session"] == [])

# ─── 15. Edge cases ──────────────────────────────────────────────────────
print("\n[15] Edge cases")

agent5 = Agent("Use the tool.", tools=[get_answer], max_tool_calls=2)
result6 = agent5.run("What is 1+1?")
check("Edge: agent handles single tool", result6.text != "")

m2 = Memory(max_messages=3)
m2.add({"role": "system", "content": "sys"})
m2.add({"role": "user", "content": "1"})
m2.add({"role": "assistant", "content": "2"})
m2.add({"role": "user", "content": "3"})
m2.add({"role": "assistant", "content": "4"})
msgs = m2.get_messages()
check("Memory: system always first", msgs[0]["role"] == "system")
check("Memory: within cap", len(msgs) <= 3)

@tool
def return_dict(x: str) -> dict:
    """Return a dict.
    Args:
        x (str): Input.
    """
    return {"key": x}

r = return_dict({"x": "test"})
check("Tool returns JSON for dict", isinstance(r, str) and "key" in r)

# ─── 16. Northwind example imports ───────────────────────────────────────
print("\n[16] Northwind example")
from miniagent.examples.northwind.schema import (
    fetch_schema, schema_prompt_block, FIELD_DESCRIPTIONS,
)
from miniagent.examples.northwind.tools import (
    odata_query, analyze_data, odata_metadata, shared_state,
)
from miniagent.examples.northwind.agents import (
    create_orchestrator_agent,
    create_data_fetcher_agent, create_data_analyst_agent,
    graph_memory,
)
check("northwind imports", True)
check("odata_query is Tool", isinstance(odata_query, Tool))
check("analyze_data is Tool", isinstance(analyze_data, Tool))
check("fetch_schema works", isinstance(fetch_schema(), dict))
check("shared_state has entities", "entities" in shared_state)
check("shared_state entities is dict", isinstance(shared_state["entities"], dict))
check("FIELD_DESCRIPTIONS has Orders", "Orders" in FIELD_DESCRIPTIONS)
check("FIELD_DESCRIPTIONS has Products", "Products" in FIELD_DESCRIPTIONS)
check("FIELD_DESCRIPTIONS has Customers", "Customers" in FIELD_DESCRIPTIONS)
check("FIELD_DESCRIPTIONS has Categories", "Categories" in FIELD_DESCRIPTIONS)
check("FIELD_DESCRIPTIONS has Order_Details", "Order_Details" in FIELD_DESCRIPTIONS)

# Verify field descriptions are complete
for entity in ["Orders", "Products", "Customers", "Employees", "Categories", "Order_Details"]:
    check(f"FIELD_DESCRIPTIONS[{entity}] non-empty", len(FIELD_DESCRIPTIONS[entity]) > 0)

# Verify schema_prompt_block includes descriptions
from miniagent.examples.northwind.schema import schema_prompt_block
block = schema_prompt_block()
check("schema_prompt_block has descriptions", "(Unique order identifier)" in block)
check("schema_prompt_block has entity names", "Orders:" in block)
check("schema_prompt_block has Products", "Products:" in block)

# ─── 17. Multi-entity shared state ───────────────────────────────────────
print("\n[17] Multi-entity shared state")

# Simulate odata_query storing entities
shared_state["entities"]["Orders"] = {"value": [{"OrderID": 1, "ShipCountry": "Germany"}]}
shared_state["entities"]["Products"] = {"value": [{"ProductID": 1, "ProductName": "Chai"}]}
check("Two entities stored", len(shared_state["entities"]) == 2)
check("Orders accessible", shared_state["entities"]["Orders"]["value"][0]["ShipCountry"] == "Germany")
check("Products accessible", shared_state["entities"]["Products"]["value"][0]["ProductName"] == "Chai")

# data_desc should reflect both
shared_state["data_desc"] = ", ".join(
    f"{k}: {len(v.get('value', []))} records" for k, v in shared_state["entities"].items()
)
check("data_desc has Orders", "Orders" in shared_state["data_desc"])
check("data_desc has Products", "Products" in shared_state["data_desc"])

# Clear entities
shared_state["entities"].clear()
check("Entities cleared", len(shared_state["entities"]) == 0)

# ─── 18. Session context injection (orchestrator) ────────────────────────
print("\n[18] Session context injection (orchestrator)")

# Build a minimal 2-agent orchestrator with shared_state
ss = {"entities": {}, "session": []}
@tool
def echo(msg: str) -> str:
    """Echo a message.
    Args:
        msg (str): Message to echo.
    """
    return msg

orch_agent = Agent(
    "Hand off to worker.",
    tools=[make_handoff_tool(["worker"])],
    hooks=[handoff_hook],
    name="orchestrator",
    max_tool_calls=2,
)
worker_agent = Agent(
    "You are a worker. Answer the question directly.",
    tools=[echo],
    name="worker",
    max_tool_calls=2,
)

orch_sess = HandoffOrchestrator(
    [orch_agent, worker_agent],
    max_handoffs=3,
    agent_names={"orchestrator": orch_agent, "worker": worker_agent},
    shared_state=ss,
)

# First question
r1 = orch_sess.run("Remember the number 42.")
check("Q1 answered", len(r1.text) > 0, r1.text[:50])
check("Session has 1 entry", len(ss["session"]) == 1)
check("Session Q1 recorded", "42" in ss["session"][0]["question"])

# Second question (context should be injected)
r2 = orch_sess.run("What number did I ask you to remember?")
check("Q2 answered", len(r2.text) > 0, r2.text[:50])
check("Session has 2 entries", len(ss["session"]) == 2)
# The worker should have seen the session context
check("Q2 references 42", "42" in r2.text, r2.text[:100])

# Reset
orch_sess.reset_session()
check("Session reset", len(ss["session"]) == 0)

# ─── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(f"  \u2717 {e}")
print("=" * 60)

sys.exit(1 if failed else 0)
