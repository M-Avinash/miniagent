# Development and Validation Approach

**Coding agent used:** Pi coding agent with a self-hosted Qwen3.8 27B LLM (llama.cpp, RTX 3090)

I worked with the coding agent as a pair programmer. I designed the architecture first (module layout, agentic loop, handoff mechanism), and then the LLM implemented each piece, which I reviewed. I validated each module with unit tests as we went, which also tied the different parts of the framework together.

I built the memory system in three stages, adding to it whenever I hit a limit. It started as plain conversation history: the system prompt, the user's questions and the agent's replies, capped at a fixed number of messages. Next, after each run, I had it save a short summary of what it was asked and what it answered to a JSONL file, and on the next run it reads those entries back. The third stage was a semantic graph. Once data is fetched, a hook pulls out the record IDs and builds a graph of what's currently in scope, and that graph goes into the prompt so the model always knows which records it's working with. I also changed the shared state to hold several data sets at once, so the analyst agent can use orders and products in the same question. The orchestrator keeps a running list of past questions and answers too, which is why a follow-up like "of those, how many had freight over 100?" works without re-fetching anything.

I validated the framework with unit tests covering every module: memory trimming (including tool-call-safe trimming), tool schema generation, sandbox timeouts, hook aborts, handoff routing, and session context injection. For the end-to-end check, I computed ground truth answers for 9 single-shot questions and 4 multi-turn conversations using pure Python queries against the OData service, with no LLM involved. The agent's answers are compared to `northwind_ground_truth.json` in `test_live.py`; I kept iterating until every answer matched exactly.

**Run the tests:**

```bash
# Offline unit tests (no network, no LLM) — safe for CI:
pytest miniagent/tests/test_unit.py

# Live integration tests (need a reachable LLM + the Northwind OData service):
MINIAGENT_RUN_LIVE=1 pytest miniagent/tests/test_live.py
```

Live tests are skipped unless `MINIAGENT_RUN_LIVE=1` is set, so the offline suite runs anywhere.
