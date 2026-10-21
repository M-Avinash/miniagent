"""orchestrator.py — multi-agent orchestration via handoffs.

Two pieces that work together:

- Marker functions: build/parse the [handoff: x] and [aborted: y] strings
  that the loop uses to signal control flow.
- HandoffOrchestrator: follows result.handoff across agents until one
  returns a result without a handoff (the final answer).

The handoff *tool* lives in tool.py (make_handoff_tool) and the handoff
*hook* lives in hooks.py (handoff_hook) — they're agent-building primitives,
not orchestration logic.

Usage:
    from miniagent import (
        Agent, make_handoff_tool, handoff_hook, HandoffOrchestrator, verbose_log,
    )

    orchestrator_agent = Agent(
        system_prompt,
        tools=[make_handoff_tool(["data_fetcher", "data_analyst"])],
        hooks=[handoff_hook, verbose_log("orchestrator")],
        name="orchestrator",
    )
    data_fetcher = Agent(..., name="data_fetcher")
    data_analyst = Agent(..., name="data_analyst")

    orch = HandoffOrchestrator([orchestrator_agent, data_fetcher, data_analyst])
    result = orch.run("How many orders were delivered to Germany last quarter?")
"""

from .events import EventType, AgentResult


# ---------------------------------------------------------------------------
# Markers — the string contract between the agent loop and the orchestrator
# ---------------------------------------------------------------------------

def make_handoff_marker(target: str) -> str:
    """Build a handoff marker string: '[handoff: agent_name]'."""
    return f"[handoff: {target}]"


def parse_handoff(text: str) -> str | None:
    """Extract the target agent name from a handoff marker, or None."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("[handoff:"):
        name = text[len("[handoff:"):].rstrip("]").strip()
        return name if name else None
    return None


def make_abort_marker(reason: str) -> str:
    """Build an abort marker string: '[aborted: reason]'."""
    return f"[aborted: {reason}]"


def parse_abort(text: str) -> str | None:
    """Extract the reason from an abort marker, or None."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("[aborted:"):
        reason = text[len("[aborted:"):].rstrip("]").strip()
        return reason if reason else None
    return None


# ---------------------------------------------------------------------------
# HandoffOrchestrator — follows the handoff chain
# ---------------------------------------------------------------------------

class HandoffOrchestrator:
    """Run a sequence of agents, following handoffs.

    Args:
        agents: A list of Agent objects. The first agent is the entry point.
        max_handoffs: Maximum number of handoffs before aborting (default 10).
        agent_names: Optional mapping of agent name → Agent object. If not
            provided, the agent's name (if set) or its index is used.
    """

    def __init__(self, agents: list, max_handoffs: int = 10, agent_names: dict = None,
                 shared_state: dict = None, on_failure=None):
        self.agents = agents
        self.max_handoffs = max_handoffs
        self.agent_names = agent_names or {}
        # shared_state: a dict shared across all agents. The orchestrator
        # maintains a 'session' key that accumulates Q&A history.
        # Agents can read/write other keys (e.g., 'entities', 'results').
        self.shared_state = shared_state if shared_state is not None else {}
        if "session" not in self.shared_state:
            self.shared_state["session"] = []  # list of {question, answer}
        # on_failure: optional callback (failed_agent, result, original_question) ->
        #   AgentResult  (accept the abort, return this result)
        #   Agent        (retry with this agent)
        #   str          (retry with this message)
        #   None         (accept the abort, return the original result)
        self.on_failure = on_failure

    def _get_agent(self, name: str) -> object:
        """Look up an agent by name. Returns None if not found.

        Checks the agent_names mapping first, then falls back to matching
        each agent's name attribute.
        """
        if name in self.agent_names:
            return self.agent_names[name]
        for agent in self.agents:
            if getattr(agent, "name", None) == name:
                return agent
        return None

    def _session_block(self) -> str:
        """Return the session context block, or empty string if no history."""
        session = self.shared_state.get("session", [])
        if not session:
            return ""
        lines = ["Previous Q&A in this session:"]
        for i, qa in enumerate(session, 1):
            lines.append(f"  Q{i}: {qa['question']}")
            lines.append(f"  A{i}: {qa['answer']}")
        return "\n".join(lines)

    def _build_user_message(self, original_question: str, previous_result: str = None) -> str:
        """Build a flat user message for an agent.

        Structure:
          Previous agent result: <immediate predecessor output>  (if any)
          Original question: <the user's actual question>
          Previous Q&A in this session: ...  (if any)
        """
        parts = []
        if previous_result:
            parts.append(f"Previous agent result: {previous_result}")
        parts.append(f"Original question: {original_question}")
        session_block = self._session_block()
        if session_block:
            parts.append(session_block)
        return "\n\n".join(parts)

    def run(self, user_message: str) -> AgentResult:
        """Run the orchestrator, following handoffs until done.

        Starts with the first agent. After each run, if the result has a
        handoff, looks up the target agent and runs it, passing the previous
        result as context. Stops when an agent returns a result without a
        handoff (the final answer) or when max_handoffs is exceeded.

        Session context (previous Q&A) is injected into every agent's user
        message automatically via shared_state['session']. Agents do not
        need to know whether this is a multi-turn conversation — the context
        is just there.

        Args:
            user_message: The initial user message to pass to the first agent.

        Returns:
            The final AgentResult from the last agent that didn't hand off.
        """
        current_agent = self.agents[0]
        original_question = user_message  # keep the clean question
        # Build the initial user message (no previous result, just the question)
        user_message = self._build_user_message(original_question)

        for _ in range(self.max_handoffs + 1):
            result = current_agent.run(user_message)

            # Failure routing: if the agent aborted, give on_failure a chance
            # to retry with a different agent or message.
            if result.aborted and self.on_failure:
                action = self.on_failure(current_agent, result, original_question)
                if isinstance(action, AgentResult):
                    return action
                elif isinstance(action, str):
                    user_message = action
                    continue  # retry same agent with new message
                elif hasattr(action, "run"):
                    current_agent = action
                    user_message = self._build_user_message(
                        original_question, previous_result=result.text,
                    )
                    continue  # retry with a different agent
                # else: None → accept the abort

            if result.handoff:
                next_agent = self._get_agent(result.handoff)
                if next_agent is None:
                    return AgentResult(
                        text=result.text,
                        turns=result.turns,
                        tool_calls=result.tool_calls,
                        aborted=True,
                        abort_reason=f"Unknown handoff target: {result.handoff}",
                    )
                current_agent = next_agent
                # Build a flat message: previous result + original question + session
                user_message = self._build_user_message(
                    original_question, previous_result=result.text,
                )
            else:
                # Record this Q&A in the session for future turns
                self.shared_state["session"].append({
                    "question": original_question,
                    "answer": result.text,
                })
                return result

        return AgentResult(
            text=result.text,
            turns=result.turns,
            tool_calls=result.tool_calls,
            aborted=True,
            abort_reason=f"Max handoffs exceeded ({self.max_handoffs})",
        )

    def reset_session(self) -> None:
        """Clear the session history (start a fresh conversation)."""
        self.shared_state["session"] = []
