"""miniagent playground — edit VARIABLES, re-run, watch components work.

    python3 -u miniagent/playground.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent.llm import LLMClient
from miniagent.tool import tool

# ============================================================
# VARIABLES — edit these, re-run
# ============================================================
QUESTION = "what is the capital of france and what is the capital of germany?"
TOOL_CHOICE = "required"      # "required" | "auto" | "none"
CHAT_PROMPT = "Say hello in three languages."
MAX_TURNS = 3                 # safety cap for the tool loop

c = LLMClient()

def section(n, title):
    print(f"\n{'=' * 62}\n{n}. {title}\n{'=' * 62}")


def dedupe_repeat(text, min_len=40):
    """Cut off a looping model: find where it starts repeating and keep the first pass.

    Local models often fail to emit a stop token, so they re-answer the same
    thing until max_tokens. We detect the loop by sliding a word-window and
    cutting at the first point a window repeats.
    """
    words = text.split()
    if len(words) < 20:
        return text
    win = 8  # words in the fingerprint window
    seen = {}  # window-start-index -> first position where that window appeared
    for i in range(len(words) - win):
        key = tuple(words[i:i + win])
        # only care about windows with enough signal (skip tiny/empty ones)
        if sum(len(w) for w in key) < min_len:
            continue
        if key in seen:
            cut = seen[key]
            return text[:cut].rstrip() + " …[repetition cut]"
        seen[key] = i
    return text

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
section(3, "Tool CALLING — the LLM decides to use the tool")
print(f"question: {QUESTION}   (tool_choice={TOOL_CHOICE})")
messages = [
    {"role": "system", "content": "You are a helpful assistant with a search tool. Use it to look up facts."},
    {"role": "user", "content": QUESTION},
]
print("[debug] turn 1: calling LLM with tools...")
try:
    resp = c.chat(messages, tools=[search_web.to_openai_schema()],
                  tool_choice=TOOL_CHOICE, max_tokens=800)
except Exception as ex:
    print(f"(error: {type(ex).__name__}: {ex})")
    resp = None

msg = resp.choices[0].message if resp else None
print(f"[debug] turn 1 done. tool_calls={bool(msg.tool_calls if msg else None)}, "
      f"content_len={len(msg.content) if msg and msg.content else 0}")

if msg and msg.tool_calls:
    tc = msg.tool_calls[0]
    print(f"-> LLM called: {tc.function.name}  args: {tc.function.arguments}")
    print("[debug] executing tool...")
    result = search_web(**json.loads(tc.function.arguments))
    print(f"[debug] tool done ({len(result)} chars)")
    messages.append(msg.model_dump())
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Loop while the model keeps calling tools (capped, like the real agent)
    for turn in range(2, MAX_TURNS + 1):
        print(f"[debug] turn {turn}: asking for final answer (tools off)...")
        final = c.chat(messages, max_tokens=400)   # no tools -> must answer
        answer = final.choices[0].message
        if answer.tool_calls:                      # model insists on more tools
            tc = answer.tool_calls[0]
            print(f"[debug] model wants another call: {tc.function.name}")
            messages.append(answer.model_dump())
            result = search_web(**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue
        print(f"\nfinal answer:\n{answer.content}")
        break
    else:
        print("\n(max turns reached — stopping)")
elif msg:
    print(f"-> NO tool call, answered directly:\n{msg.content}")

print("\nEdit the VARIABLES at the top and re-run.")
