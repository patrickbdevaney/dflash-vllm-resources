#!/usr/bin/env python3
# bench.py — DFlash decode benchmark (tok/s e2e + acceptance length tau) per coding task.
# Usage: python3 bench.py --url http://localhost:8001 --tasks all|dijkstra[,lru,...] --runs 3
import urllib.request, json, time, re, argparse, statistics, sys

TASKS = {
    "sorting":  ("Write quicksort and mergesort in Python with tests.", 768),
    "lru":      ("Implement a Python LRU cache class with get/put, O(1), full docstrings.", 768),
    "dijkstra": ("Write a Python implementation of Dijkstra shortest path with a heap, type hints, docstrings.", 768),
    "mixed":    ("You are a coding assistant. Implement a thread-safe bounded blocking queue in Python, "
                 "then briefly explain the concurrency strategy.", 768),
}

def get_metrics(url):
    try:
        t = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
        g = lambda k: float(re.search(rf'vllm:{k}\{{[^}}]*}} ([0-9.eE+]+)', t).group(1))
        return g('spec_decode_num_drafts_total'), g('spec_decode_num_accepted_tokens_total')
    except Exception:
        return None, None

def ask(url, prompt, mt, model="/model"):
    p = {"model": model, "messages": [{"role": "user", "content": prompt}],
         "max_tokens": mt, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    el = time.perf_counter() - t0
    return r["usage"]["completion_tokens"], el

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="/model")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    names = list(TASKS) if a.tasks == "all" else [t.strip() for t in a.tasks.split(",")]
    ask(a.url, "Say hi.", 8, a.model)  # warm
    print(f"=== bench {a.label} | url={a.url} | runs={a.runs} ===")
    print(f"{'task':10s} {'tok/s(e2e)':>12s} {'tau':>6s} {'tokens':>7s}")
    summary = {}
    for name in names:
        prompt, mt = TASKS[name]
        tps_list, taus = [], []
        for _ in range(a.runs):
            d0 = get_metrics(a.url); n, el = ask(a.url, prompt, mt, a.model); d1 = get_metrics(a.url)
            tps_list.append(n / el)
            if d0[0] is not None and d1[0] - d0[0] > 0:
                taus.append((d1[1] - d0[1]) / (d1[0] - d0[0]) + 1)
        med = statistics.median(tps_list)
        tau = statistics.median(taus) if taus else float("nan")
        summary[name] = (med, tau)
        print(f"{name:10s} {med:12.1f} {tau:6.2f} {mt:7d}")
    print("SUMMARY_JSON", json.dumps({k: {"tok_s": round(v[0],1), "tau": round(v[1],2)} for k,v in summary.items()}))

if __name__ == "__main__":
    main()
