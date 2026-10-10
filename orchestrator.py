"""Handoff orchestrator — the logic that decides which agent runs next.

Agents can hand off to each other by setting `result.handoff = "agent_name"`.
The orchestrator just follows the handoffs until an agent returns a result
without a handoff (or the max-handoff limit is hit).

This is the simplest form of multi-agent orchestration: a loop that follows
agent-to-agent handoffs, just a loop.
"""

from .events import AgentResult


class HandoffOrchestrator:
    """Run a sequence of agents, following handoffs.

    Args:
        agents: A list of Agent objects. The first agent is the entry point.
        max_handoffs: Maximum number of handoffs before aborting (default 10).
        agent_names: Optional mapping of agent name → Agent object. If not
            provided, the agent's name (if set) or its index is used.

    Usage:
        orchestrator = HandoffOrchestrator([
            orchestrator_agent,
            data_fetcher_agent,
            data_analyst_agent,
        ])
        result = orchestrator.run("How many orders were delivered to Germany last quarter?")
    """

    def __init__(self, agents: list, max_handoffs: int = 10, agent_names: dict = None):
        self.agents = agents
        self.max_handoffs = max_handoffs
        self.agent_names = agent_names or {}

    def _get_agent(self, name: str) -> object:
        """Look up an agent by name. Returns None if not found."""
        if name in self.agent_names:
            return self.agent_names[name]
        # Fallback: try to find by the agent's name attribute
        for agent in self.agents:
            if getattr(agent, "name", None) == name:
                return agent
        return None

    def run(self, user_message: str) -> AgentResult:
        """Run the orchestrator, following handoffs until done.

        Args:
            user_message: The initial user message to pass to the first agent.

        Returns:
            The final AgentResult from the last agent that didn't hand off.
        """
        current_agent = self.agents[0]
        handoff_count = 0

        for _ in range(self.max_handoffs + 1):
            result = current_agent.run(user_message)

            # If the agent wants to hand off, follow the handoff
            if result.handoff:
                handoff_count += 1
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
                # Pass the current result as context for the next agent
                user_message = f"Previous agent result: {result.text}\n\nOriginal question: {user_message}"
            else:
                # No handoff — this is the final result
                return result

        # Hit the max-handoff limit
        return AgentResult(
            text=result.text,
            turns=result.turns,
            tool_calls=result.tool_calls,
            aborted=True,
            abort_reason=f"Max handoffs exceeded ({self.max_handoffs})",
        )
