import json
import urllib.request
import time
import sys
import argparse

API_BASE = "http://192.168.86.201:52415"
MODEL_ID = "mlx-community/MiniMax-M2.5-5bit"

def get_large_prompt():
    prompt = "You are a senior systems engineer. Provide a very detailed, step-by-step guide to building a scalable, fault-tolerant distributed system from scratch. Include code examples, architecture diagrams described in text, and comprehensive explanations of consensus algorithms, load balancing, and database sharding. " * 100
    return prompt

def run_inference(prompt, max_tokens=4096):
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    t_start = time.time()
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if b"[DONE]" in line:
                    break
                if b'"content"' in line or b'"reasoning_content"' in line:
                    tokens += 1
    except Exception as e:
        print(f"Error during inference: {e}")
        return False
    
    elapsed = time.time() - t_start
    print(f"Completed {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tps)")
    return True

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
        # Add massive context to grow KV quickly
        growth_prompt = f"\n\n[KV STRESS BLOCK {iteration}]\n{massive_block}\n\nContinue the guide."
        conversation_history[-1]["content"] += growth_prompt
        
        full_response = ""
        success = False
        
        try:
            # Record start for TPS calculation
            start_time = time.time()
            token_count = 0
            
            payload = json.dumps({
                "model": MODEL_ID,
                "messages": conversation_history,
                "max_tokens": 4096, # MAX OUTPUT to grow KV fast
                "stream": True,
                "temperature": 0.7
            })
            
            req = urllib.request.Request(f"{API_BASE}/v1/chat/completions", data=payload.encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
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
                                    # Don't print tokens to stdout to avoid terminal spam, just print dots
                                    if token_count % 10 == 0:
                                        print(".", end='', flush=True)
                                    full_response += token
                                    token_count += 1
                            except:
                                continue
            
            if token_count > 0:
                elapsed = time.time() - start_time
                print(f"\n\n[Stats] Generated {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} TPS)")
                
                # Append response to history
                conversation_history.append({"role": "assistant", "content": full_response})
                # Add a new user prompt for the next turn
                conversation_history.append({"role": "user", "content": "Keep going with the next architectural layer."})
                success = True
                
        except Exception as e:
            print(f"\nHyper-Inference failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if success:
            iteration += 1
            time.sleep(1)
