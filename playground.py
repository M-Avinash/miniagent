"""miniagent playground — edit VARIABLES, re-run, watch components work.

    python3 -u miniagent/playground.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent.llm import LLMClient
from miniagent.tool import tool
from miniagent.agent import Agent

# ============================================================
# VARIABLES — edit these, re-run
# ============================================================
QUESTION = "what is the capital of france and what is the capital of germany?"
CHAT_PROMPT = "Say hello in three languages."
MAX_TOOL_CALLS = 5            # safety cap for the agent's tool loop

c = LLMClient()

def section(n, title):
    print(f"\n{'=' * 62}\n{n}. {title}\n{'=' * 62}")


# ------------------------------------------------------------
section(1, "LLM client — live chat")
print(f"model={c.model}  endpoint={c._client.base_url}")
try:
    resp = c.chat([{"role": "user", "content": CHAT_PROMPT}], max_tokens=200)
    print(f"Q: {CHAT_PROMPT}")
    print(f"A: {resp.choices[0].message.content}")
except Exception as ex:
    print(f"(server not reachable: {type(ex).__name__})")

# ------------------------------------------------------------
section(2, "Tool — a function becomes a tool")

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.
    Args:
        query (str): The search query.
        max_results (int): Max results to return.
    """
    r = c.chat([
        {"role": "system", "content": f"Search engine. Answer in at most {max_results} bullet points. Be brief."},
        {"role": "user", "content": query},
    ], max_tokens=300)
    return r.choices[0].message.content

try:
    print(search_web(query="quantum computing", max_results=2)[:200])
except Exception as ex:
    print(f"(server not reachable: {type(ex).__name__})")

# ------------------------------------------------------------
section(3, "Tool CALLING — the agent loop (unified, one source of truth)")
print(f"question: {QUESTION}")
print("(tools stay available every turn; turn 1 forces a tool, later turns are auto;")
print(" the model stops when it decides to answer — no more ripping tools away.)")

agent = Agent(
    "You are a helpful assistant with a search tool. Use it to look up facts.",
    tools=[search_web],
    max_tool_calls=MAX_TOOL_CALLS,
)
try:
    result = agent.run(QUESTION)
except Exception as ex:
    print(f"(error: {type(ex).__name__}: {ex})")
    result = None

if result:
    print(f"\nturns: {result.turns}   tool_calls: {len(result.tool_calls)}   aborted: {result.aborted}")
    for tc in result.tool_calls:
        print(f"   -> {tc['name']}({tc['args']})")
    if result.aborted:
        print(f"(aborted: {result.abort_reason})")
    print(f"\nfinal answer:\n{result.text}")

print("\nEdit the VARIABLES at the top and re-run.")
