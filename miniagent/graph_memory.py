"""graph_memory.py — semantic graph memory for relational schemas.

Tracks entity keys and navigation paths across a multi-turn conversation.
A Python dict of entity keys plus a path list. A hook populates it after
each data fetch; another hook injects the active context into the LLM
prompt before each call.

Usage:
    from miniagent import GraphMemory, graph_memory_hook, graph_injection_hook

    graph = GraphMemory()
    agent = Agent(
        system_prompt,
        tools=[odata_query],
        hooks=[
            graph_memory_hook(graph, {"Order": "OrderID", "Customer": "CustomerID"}),
            graph_injection_hook(graph),
            verbose_log("agent"),
        ],
    )
"""

import json


class GraphMemory:
    """A graph of entity keys and navigation paths.

    Nodes are keyed by (entity_type, key). The path is an ordered list of
    the entities the conversation has navigated through.

    Example:
        graph.add_record("Customer", {"CustomerID": "ALFKI", ...}, "CustomerID")
        graph.add_record("Order", {"OrderID": 10248, ...}, "OrderID")
        graph.focus_path()
        # -> "Customer('ALFKI') -> Order('10248')"
    """

    def __init__(self):
        self.nodes = {}
        self.path = []

    def add_record(self, entity_type: str, record: dict, key_field: str) -> None:
        """Add a record to the graph.

        Args:
            entity_type: The entity name (e.g. "Customer", "Order").
            record: The record dict (e.g. {"CustomerID": "ALFKI", ...}).
            key_field: The field name that holds the primary key.
        """
        key = record.get(key_field)
        if key is None:
            return
        key_str = str(key)
        self.nodes.setdefault(entity_type, {})[key_str] = record
        path_entry = f"{entity_type}('{key_str}')"
        if not self.path or self.path[-1] != path_entry:
            self.path.append(path_entry)

    def focus_path(self) -> str:
        """Return the current navigation path as a string."""
        return " -> ".join(self.path) if self.path else "(empty)"

    def active_keys(self) -> dict:
        """Return the active entity keys as a dict of {entity_type: [keys]}."""
        return {et: list(keys.keys()) for et, keys in self.nodes.items()}

    def get_record(self, entity_type: str, key) -> dict:
        """Get a record by entity type and key. Returns None if not found."""
        return self.nodes.get(entity_type, {}).get(str(key))

    def prompt_block(self) -> str:
        """Return the compressed prompt block for context injection.

        Returns an empty string if the graph is empty.
        """
        if not self.path:
            return ""
        return (
            "[SEMANTIC GRAPH MEMORY ACTIVE]\n"
            f"Current Data Focus Path: {self.focus_path()}\n"
            f"Available Entity Keys in Memory: {self.active_keys()}\n"
        )

    def clear(self) -> None:
        """Clear the graph (start a fresh conversation)."""
        self.nodes = {}
        self.path = []

    def __repr__(self) -> str:
        if not self.path:
            return "GraphMemory(empty)"
        return f"GraphMemory(path={self.focus_path()!r})"


def graph_memory_hook(graph: GraphMemory, entity_key_map: dict):
    """Create a hook that populates the graph after a data-fetch tool call.

    On each TOOL_RESULT, parses the JSON result and adds each record's
    primary key to the graph.

    Args:
        graph: The GraphMemory instance to populate.
        entity_key_map: Maps entity type to its key field, e.g.
            {"Order": "OrderID", "Customer": "CustomerID"}.

    Returns:
        A hook function to attach to an agent.
    """
    def hook(event):
        if event.type.name != "TOOL_RESULT":
            return event
        result = event.tool_result
        if not isinstance(result, str):
            return event
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return event
        records = data.get("value", []) if isinstance(data, dict) else []
        if not isinstance(records, list):
            return event
        for record in records:
            if not isinstance(record, dict):
                continue
            for entity_type, key_field in entity_key_map.items():
                if key_field in record:
                    graph.add_record(entity_type, record, key_field)
        return event

    return hook


def graph_injection_hook(graph: GraphMemory):
    """Create a hook that injects the graph context into the LLM prompt.

    On each LLM_CALL, appends the graph's prompt block to the system
    message, giving the LLM explicit entity keys and the navigation path.

    Args:
        graph: The GraphMemory instance to read from.

    Returns:
        A hook function to attach to an agent.
    """
    def hook(event):
        if event.type.name != "LLM_CALL":
            return event
        block = graph.prompt_block()
        if not block:
            return event
        messages = event.messages
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"] + "\n\n" + block
            event.messages = messages
        return event

    return hook
