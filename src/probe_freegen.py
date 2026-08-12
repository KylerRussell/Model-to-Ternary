#!/usr/bin/env python
"""probe_freegen.py — scored free-gen A/B battery against a running llama-server (M5 ground truth).
Measures the things the teacher-forced gate is BLIND to: sentence/token repetition loops, and whether the
model COMMITS (closes </think> + emits a non-empty answer) vs over-thinks. Run per served model, compare.

  PORT=8137 ./.venv/bin/python src/probe_freegen.py            # thinking-on + thinking-off, temp 0 and 0.6
"""
import os, re, json, urllib.request

PORT = int(os.environ.get("PORT", "8137"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    "What is the capital of France?", "Write a haiku about the ocean.",
    "What is 17 times 3?", "Explain photosynthesis in two sentences.",
    "If a train travels 60 km at 40 km/h then 90 km at 60 km/h, what is the average speed?",
    "What is the remainder when 7^100 is divided by 13?", "Name three renewable energy sources.",
    "Write a short poem about autumn.", "How many trailing zeros are in 100! ?",
    "Recommend a book and say why in one sentence.", "What causes the seasons?",
    "A farmer has chickens and cows totaling 30 heads and 74 legs. How many of each?",
]


def has_loop(text, n=5):
    """repeated n-gram (word-level) or a verbatim-repeated sentence => degeneration loop."""
    words = text.split()
    seen = {}
    for i in range(len(words) - n):
        g = " ".join(words[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= 3:               # same 5-gram 3+ times
            return True
    sents = [s.strip() for s in re.split(r'[.\n!?]+', text) if len(s.strip()) > 8]
    sc = {}
    for s in sents:
        sc[s] = sc.get(s, 0) + 1
        if sc[s] >= 3:
            return True
    return False


def call(prompt, think, temp, max_tokens=600):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": temp,
            "max_tokens": max_tokens, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": think}}
    if temp > 0:
        body.update(top_p=0.95, top_k=20)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    ch = d["choices"][0]
    m = ch["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), ch.get("finish_reason")


def run_mode(think, temp):
    loops = commits = 0
    n = len(PROMPTS)
    for p in PROMPTS:
        ans, rc, fin = call(p, think, temp)
        full = rc + "\n" + ans
        looped = has_loop(full)
        loops += looped
        if think:
            committed = (len(ans.strip()) >= 2 and not has_loop(ans))     # closed </think> + real answer
        else:
            committed = (fin == "stop" and not looped)
        commits += committed
    return loops, commits, n


print(f"probing :{PORT}")
for think in (True, False):
    for temp in (0.0, 0.6):
        loops, commits, n = run_mode(think, temp)
        tag = f"think={'on ' if think else 'off'} temp={temp}"
        print(f"  {tag}:  loop_rate {loops}/{n}={100*loops/n:.0f}%   "
              f"{'commit' if think else 'clean-stop'}_rate {commits}/{n}={100*commits/n:.0f}%")
