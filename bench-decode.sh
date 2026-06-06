#!/bin/bash
# bench-decode.sh — DFlash decode-speed benchmark (run AFTER verify-dflash.sh passes)
set -euo pipefail
BASE_URL="${1:-http://localhost:8001}"
echo "=== Decode benchmark: DFlash k=16 on Qwen3.6-35B-A3B-NVFP4 | URL=$BASE_URL ==="
BASE_URL="$BASE_URL" python3 << 'PYEOF'
import urllib.request, json, time, statistics, os
BASE_URL=os.environ['BASE_URL']; N_RUNS=5
tasks=[
 {"name":"Short code (256)","prompt":"Write a Python binary search tree with insert, search, and inorder traversal.","max_tokens":256},
 {"name":"Long code (512)","prompt":"Write a complete Python min-heap priority queue with push, pop, peek, __len__, full type hints and docstrings.","max_tokens":512},
 {"name":"Agentic tool-call (256)","prompt":"You are an AI assistant. The user wants to search the web for 'NVIDIA Jetson Thor benchmarks'. Respond with a tool call in JSON, then a brief answer.","max_tokens":256},
]
for t in tasks:
    print(f"--- {t['name']} ---"); times=[]; toks=[]
    for i in range(N_RUNS):
        p=json.dumps({"model":"/model","messages":[{"role":"user","content":t["prompt"]}],
                      "max_tokens":t["max_tokens"],"temperature":0,"thinking_budget":0}).encode()
        req=urllib.request.Request(f"{BASE_URL}/v1/chat/completions",data=p,headers={"Content-Type":"application/json"})
        t0=time.perf_counter()
        with urllib.request.urlopen(req,timeout=120) as r: resp=json.loads(r.read())
        e=time.perf_counter()-t0; n=resp.get("usage",{}).get("completion_tokens",0)
        if n>0: times.append(e); toks.append(n)
    if times:
        tps=[a/b for a,b in zip(toks,times)]
        print(f"  avg tokens={statistics.mean(toks):.0f} avg elapsed={statistics.mean(times):.2f}s")
        print(f"  avg tok/s={statistics.mean(tps):.1f} median={statistics.median(tps):.1f} stdev={statistics.stdev(tps) if len(tps)>1 else 0:.1f}")
    print()
print("=== Reference baselines ===")
print("  DFlash k=16 NVFP4 target: ~60-90 tok/s effective (acceptance ~2.7-4.4 tok/step)")
print("  MTP-2 NVFP4 Thor baseline: ~37-70 tok/s effective")
print("  If below MTP-2: try num_speculative_tokens: 7 (block 8) and re-test.")
PYEOF
echo "=== Benchmark complete ==="
