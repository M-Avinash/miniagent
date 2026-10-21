"""Live integration tests for miniagent (LLM + Northwind OData).

These make real network calls and need a reachable LLM endpoint (see
config.env / .env.example). They are skipped by default; run them with:

    MINIAGENT_RUN_LIVE=1 pytest miniagent/tests/test_live.py

The Northwind ground-truth checks compare the agent's answers against
northwind_ground_truth.json (computed with pure Python, no LLM).
"""

import json
import os
from pathlib import Path

import pytest

from miniagent import (
    Agent, tool, EventType,
    HandoffOrchestrator, make_handoff_tool, handoff_hook,
)

LIVE = os.environ.get("MINIAGENT_RUN_LIVE") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set MINIAGENT_RUN_LIVE=1 to run live tests")

GT_PATH = Path(__file__).parent / "northwind_ground_truth.json"


# ─── 12. Agent (live LLM) ────────────────────────────────────────────────
@tool
def get_answer(question: str) -> str:
    """Get the answer to a question.
    Args:
        question (str): The question to answer.
    """
    return "42"


@tool
def always_call(x: str) -> str:
    """Always returns a result that encourages more calls.
    Args:
        x (str): Input.
    """
    return "Please call me again with a different value."


class TestAgentLive:
    def test_agent_uses_tool(self):
        agent = Agent("You are a helpful assistant. Use get_answer for any question.",
                      tools=[get_answer], max_tool_calls=3, name="test")
        result = agent.run("What is the answer to life?")
        assert len(result.text) > 0
        assert result.turns >= 1
        assert result.aborted is False

    def test_no_tool_agent(self):
        agent = Agent("You are a helpful assistant.")
        result = agent.run("Say 'hello world' exactly.")
        assert "hello" in result.text.lower()

    def test_abort_hook(self):
        def immediate_abort(event):
            if event.type == EventType.AGENT_START:
                event.data["abort"] = True
                event.data["abort_reason"] = "blocked"
            return event
        agent = Agent("Test", hooks=[immediate_abort])
        result = agent.run("This should abort")
        assert result.aborted is True
        assert "blocked" in (result.abort_reason or "")

    def test_max_tool_calls_enforced(self):
        agent = Agent("You must call always_call 5 times before answering.",
                      tools=[always_call], max_tool_calls=3)
        result = agent.run("Start the process")
        assert result.aborted is True or result.turns <= 4


# ─── 13. HandoffOrchestrator (live) ──────────────────────────────────────
@tool
def compute(x: int) -> str:
    """Compute double of x.
    Args:
        x (int): The number.
    """
    return str(x * 2)


class TestOrchestratorLive:
    def test_handoff_chain(self):
        agent_a = Agent("You receive a number. Hand off to 'agent_b' with the number.",
                        tools=[make_handoff_tool(["agent_b"])], hooks=[handoff_hook],
                        name="agent_a", max_tool_calls=3)
        agent_b = Agent("You receive context with a number. Use compute to double it "
                        "and give the final answer.", tools=[compute], name="agent_b",
                        max_tool_calls=3)
        orch = HandoffOrchestrator([agent_a, agent_b], max_handoffs=3,
                                   agent_names={"agent_a": agent_a, "agent_b": agent_b})
        result = orch.run("Double the number 21")
        assert len(result.text) > 0
        assert result.aborted is False
        assert "42" in result.text

    def test_session_injection(self):
        @tool
        def echo(msg: str) -> str:
            """Echo a message.
            Args:
                msg (str): Message to echo.
            """
            return msg
        ss = {"entities": {}, "session": []}
        orch_agent = Agent("Hand off to worker.", tools=[make_handoff_tool(["worker"])],
                           hooks=[handoff_hook], name="orchestrator", max_tool_calls=2)
        worker_agent = Agent("You are a worker. Answer the question directly.",
                             tools=[echo], name="worker", max_tool_calls=2)
        orch = HandoffOrchestrator([orch_agent, worker_agent], max_handoffs=3,
                                   agent_names={"orchestrator": orch_agent, "worker": worker_agent},
                                   shared_state=ss)
        r1 = orch.run("Remember the number 42.")
        assert len(r1.text) > 0
        assert len(ss["session"]) == 1
        assert "42" in ss["session"][0]["question"]
        r2 = orch.run("What number did I ask you to remember?")
        assert len(r2.text) > 0
        assert len(ss["session"]) == 2
        assert "42" in r2.text
        orch.reset_session()
        assert len(ss["session"]) == 0


# ─── 16. Northwind example (live OData) ──────────────────────────────────
class TestNorthwindLive:
    def test_fetch_schema(self):
        from miniagent.examples.northwind.schema import fetch_schema
        schema = fetch_schema()
        assert isinstance(schema, dict)
        assert len(schema) > 0

    def test_schema_prompt_block(self):
        from miniagent.examples.northwind.schema import fetch_schema, schema_prompt_block
        fetch_schema()
        block = schema_prompt_block()
        assert "Orders:" in block
        assert "Products" in block


# ─── Ground truth: agent answers must match precomputed values ───────────
def _build_northwind_orchestrator():
    from miniagent.examples.northwind.tools import shared_state
    from miniagent.examples.northwind.agents import (
        create_orchestrator_agent, create_data_fetcher_agent,
        create_data_analyst_agent, graph_memory,
    )
    from miniagent.examples.northwind.schema import fetch_schema
    fetch_schema()
    orchestrator = create_orchestrator_agent()
    data_fetcher = create_data_fetcher_agent()
    data_analyst = create_data_analyst_agent()
    orch = HandoffOrchestrator(
        [orchestrator, data_fetcher, data_analyst],
        max_handoffs=5,
        agent_names={"orchestrator": orchestrator, "data_fetcher": data_fetcher,
                     "data_analyst": data_analyst},
        shared_state=shared_state,
    )
    return orch, data_analyst, shared_state, graph_memory


def _answer_contains(text: str, *needles) -> bool:
    """True if every needle appears in text (numbers matched loosely)."""
    low = text.lower()
    for n in needles:
        if isinstance(n, float):
            # match the number or a rounded form
            if str(n) not in text and f"{n:.2f}" not in text and f"{n:.0f}" not in text:
                return False
        elif str(n) not in text:
            return False
    return True


class TestNorthwindGroundTruth:
    """Compare agent answers to northwind_ground_truth.json.

    These are the end-to-end checks that docs/validation.md describes.
    They are intentionally a small, high-signal subset (the single-shot
    questions with unambiguous numeric answers).
    """

    @pytest.fixture(scope="class")
    def gt(self):
        return json.loads(GT_PATH.read_text())

    @pytest.fixture(scope="class")
    def orch(self):
        o, analyst, shared, graph = _build_northwind_orchestrator()
        return o, analyst, shared, graph

    def test_q1_germany_q1_1997(self, gt, orch):
        o, analyst, shared, graph = orch
        shared["entities"].clear()
        graph.clear()
        analyst.reset_analyze_attempts()
        r = o.run("How many orders were shipped to Germany in Q1 1997?")
        expected = gt["single_shot"]["q1_germany_q1_1997"]["count"]
        assert _answer_contains(r.text, str(expected)), r.text

    def test_q2_most_orders_1997(self, gt, orch):
        o, analyst, shared, graph = orch
        shared["entities"].clear()
        graph.clear()
        analyst.reset_analyze_attempts()
        r = o.run("Which country received the most orders in 1997?")
        expected = gt["single_shot"]["q2_most_orders_1997"]["country"]
        assert expected.lower() in r.text.lower(), r.text

    def test_q7_beverages_out_of_stock(self, gt, orch):
        o, analyst, shared, graph = orch
        shared["entities"].clear()
        graph.clear()
        analyst.reset_analyze_attempts()
        r = o.run("How many products in the Beverages category are out of stock?")
        expected = gt["single_shot"]["q7_beverages_out_of_stock"]["count"]
        assert _answer_contains(r.text, str(expected)), r.text
