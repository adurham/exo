import json
import urllib.request
import time
import sys

API_BASE = "http://192.168.86.201:52415"
MODEL_ID = "mlx-community/MiniMax-M2.5-5bit"

def get_large_prompt():
    prompt = "You are a senior systems engineer. Provide a very detailed, step-by-step guide to building a scalable, fault-tolerant distributed system from scratch. Include code examples, architecture diagrams described in text, and comprehensive explanations of consensus algorithms, load balancing, and database sharding. " * 100
    return prompt

if __name__ == "__main__":
    print("Starting HYPER-GROWTH KV cache stress test...")
    base_prompt = get_large_prompt()
    conversation_history = [
        {"role": "user", "content": base_prompt}
    ]
    iteration = 1
    
    # Pre-generate a massive technical block to append each time
    massive_block = ("This is an extended architectural deep-dive into distributed systems. " * 100) + "\n"
    massive_block = massive_block * 5 # Roughly 5000+ words
    
    while True:
        print(f"\n--- Hyper-Iteration {iteration} ---")
        growth_prompt = f"\n\n[KV STRESS BLOCK {iteration}]\n{massive_block}\n\nContinue the guide."
        conversation_history[-1]["content"] += growth_prompt
        
        full_response = ""
        success = False
        
        try:
            start_time = time.time()
            token_count = 0
            
            payload = json.dumps({
                "model": MODEL_ID,
                "messages": conversation_history,
                "max_tokens": 4096,
                "stream": True,
                "temperature": 0.7
            })
            
            # Use 600 second timeout for the massive prefill
            req = urllib.request.Request(f"{API_BASE}/v1/chat/completions", data=payload.encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as response:
                for line in response:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data: "):
                            data_content = line_str[6:]
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_content)
                                token = chunk['choices'][0]['delta'].get('content', '')
                                if token:
                                    if token_count % 10 == 0:
                                        print(".", end='', flush=True)
                                    full_response += token
                                    token_count += 1
                            except:
                                continue
            
            if token_count > 0:
                elapsed = time.time() - start_time
                print(f"\n\n[Stats] Generated {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} TPS)")
                
                conversation_history.append({"role": "assistant", "content": full_response})
                conversation_history.append({"role": "user", "content": "Keep going with the next architectural layer."})
                success = True
                
        except Exception as e:
            print(f"\nHyper-Inference failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if success:
            iteration += 1
            time.sleep(1)