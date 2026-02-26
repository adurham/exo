import time
import requests
import json
import sys

URL = "http://adams-mac-studio-m4-1.local:52415/v1/chat/completions"
MODEL = "mlx-community/MiniMax-M2.5-5bit"
DURATION_MINUTES = 30
END_TIME = time.time() + (DURATION_MINUTES * 60)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Explain quantum entanglement in 2 sentences."}],
    "max_tokens": 100,
    "stream": False
}

print(f"Starting 30-minute benchmark against {URL}")
print(f"Target model: {MODEL}")

success_count = 0
failure_count = 0

try:
    while time.time() < END_TIME:
        try:
            start_req = time.time()
            response = requests.post(URL, json=payload, timeout=120)
            elapsed = time.time() - start_req
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    success_count += 1
                    print(f"[{time.strftime('%H:%M:%S')}] Success ({elapsed:.2f}s): {content[:50].strip()}...")
                else:
                    failure_count += 1
                    print(f"[{time.strftime('%H:%M:%S')}] FAILURE: Empty content received.")
            else:
                failure_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] FAILURE: Status code {response.status_code}. Response: {response.text}")
                
        except Exception as e:
            failure_count += 1
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        
        # Small delay between requests
        time.sleep(2)

except KeyboardInterrupt:
    print("\nBenchmark interrupted by user.")

print("\n--- Benchmark Results ---")
print(f"Total Successes: {success_count}")
print(f"Total Failures: {failure_count}")
print(f"Duration: {DURATION_MINUTES} minutes")
