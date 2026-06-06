#!/bin/bash
# verify-dflash.sh — run from a 2nd terminal once the server prints "Application startup complete"
set -euo pipefail
BASE_URL="${1:-http://localhost:8001}"
echo "=== DFlash Verification Suite ==="

echo "--- Test 1: Coherence (short coding task, T=0) ---"
curl -sf "${BASE_URL}/v1/chat/completions" -H "Content-Type: application/json" -d '{
    "model":"/model",
    "messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number using memoization. Return only the code."}],
    "max_tokens":256,"temperature":0,"thinking_budget":0}' | python3 -c "
import json,sys
d=json.load(sys.stdin); msg=d['choices'][0]['message']; c=msg.get('content') or ''
print('output_tokens:', d.get('usage',{}).get('completion_tokens','?'))
print('head:', c[:200])
print('PASS' if 'def' in c and 'return' in c else 'FAIL: no function body')"

echo ""; echo "--- Test 2: spec_decode acceptance metrics ---"
M=$(curl -sf "${BASE_URL}/metrics" 2>/dev/null || echo "")
if echo "$M" | grep -q "spec_decode"; then echo "$M" | grep "spec_decode" | head -20
else echo "no spec_decode metrics yet — use decode speed as proxy"; fi

echo ""; echo "--- Test 3: Decode speed (512-tok code gen, T=0) ---"
python3 -c "
import urllib.request,json,time
p=json.dumps({'model':'/model','messages':[{'role':'user','content':'Write a complete merge sort implementation in Python with full type hints and docstrings.'}],'max_tokens':512,'temperature':0,'thinking_budget':0}).encode()
r=urllib.request.Request('${BASE_URL}/v1/chat/completions',data=p,headers={'Content-Type':'application/json'})
t=time.perf_counter()
resp=json.loads(urllib.request.urlopen(r).read()); e=time.perf_counter()-t
n=resp.get('usage',{}).get('completion_tokens',0)
print(f'tokens={n} elapsed={e:.2f}s tok/s(e2e)={n/e if e>0 else 0:.1f}')
print('DFlash k=16 expected on Thor: ~70-120 tok/s | MTP-2 baseline: ~37-70 tok/s')"

echo ""; echo "--- Test 4: thinking_budget + reasoning ---"
curl -sf "${BASE_URL}/v1/chat/completions" -H "Content-Type: application/json" -d '{
    "model":"/model","messages":[{"role":"user","content":"Prove there are infinitely many primes."}],
    "max_tokens":1024,"thinking_budget":128}' | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
rc=m.get('reasoning_content') or ''; a=m.get('content') or ''
print('reasoning_words~', len(rc.split()), '| answer_head:', a[:80])
print('PASS' if len(rc)>0 and len(a)>20 else 'FAIL')"

echo ""; echo "=== INTERPRETING ==="
echo "  tok/s << 40   -> DFlash fell back to AR (check logs for DFlashProposer init)"
echo "  tok/s 70-120  -> DFlash working well"
echo "  garbled       -> cudagraph size not multiple of 16; re-run with 'eager' first"
