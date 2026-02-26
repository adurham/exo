import urllib.request
import json
import sys

API_BASE = "http://192.168.86.201:52415"
MODEL_ID = "mlx-community/MiniMax-M2.5-5bit"
prompt = "Explain in one short paragraph how distributed systems achieve fault tolerance."

payload = json.dumps({
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
    "max_tokens": 100,
}).encode()

req = urllib.request.Request(
    f"{API_BASE}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

try:
    print("Sending request...")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        reasoning = result.get('choices', [{}])[0].get('message', {}).get('reasoning_content', '')
        print("\n--- OUTPUT ---")
        if reasoning:
            print(f"<think>{reasoning}</think>")
        print(content)
        print("--------------")
except Exception as e:
    print(f"Error: {e}")
