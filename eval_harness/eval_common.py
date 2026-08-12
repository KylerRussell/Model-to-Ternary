"""Shared eval client: concurrent requests load-balanced across N served endpoints.

Used by math_eval.py and mcq_eval.py so a single config drives both the ternary phase
(two endpoints, one per GPU) and the FP phase (one tensor-split endpoint).
"""
import json, threading, urllib.request, concurrent.futures as cf


def chat(endpoint, content, max_tokens=12288, temp=0.6, top_p=0.95, timeout=3000, template_kwargs=None):
    payload = {"messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": temp, "top_p": top_p}
    if template_kwargs:                       # e.g. {"enable_thinking": False} for non-CoT MCQ
        payload["chat_template_kwargs"] = template_kwargs
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason")


def run_concurrent(n, endpoints, concurrency, worker, label=""):
    """Run worker(i, endpoint) for i in range(n), <=concurrency in flight, round-robin endpoints.
    Returns results list (index-aligned). Robust to per-task errors (records None)."""
    results = [None] * n
    done = [0]
    lock = threading.Lock()

    def run(i):
        ep = endpoints[i % len(endpoints)]
        try:
            results[i] = worker(i, ep)
        except Exception as e:
            results[i] = {"error": str(e)}
        with lock:
            done[0] += 1
            if done[0] % max(1, n // 50) == 0 or done[0] == n:
                print(f"   [{label}] {done[0]}/{n}", flush=True)
        return i

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(run, range(n)))
    return results
