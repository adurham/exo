#!/usr/bin/env python3
"""
Collect real-generation top1-vs-top2 margin distribution.
Margin_i = top1_logprob - top2_logprob.
"""

import json
import os
import time
import httpx

API_URL = "http://192.168.86.201:52415/v1/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
OUTPUT_DIR = "/Users/adam.durham/repos/exo/tmp/p05-lmhead-mxfp8-20260830/real_margins"

def collect_margins(label, prompt, target_tokens=1000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": target_tokens,
        "logprobs": True,
        "top_logprobs": 2,
    }
    
    print(f"Collecting margins for {label}...")
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            resp = client.post(API_URL, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"Request failed for {label}: {e}")
        return None

    # Extract logprobs.content array
    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
    if not logprobs_data:
        print(f"No logprobs returned for {label}. STOPPING as per instructions.")
        return "BLOCKER"

    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
    
    positions = []
    for i, item in enumerate(logprobs_data):
        token = item.get("token")
        top1_lp = item.get("logprob")
        
        top_lps = item.get("top_logprobs", [])
        top2_lp = -float('inf')
        
        # Logprobs are typically returned as a list of (token, logprob) tuples
        if isinstance(top_lps, list) and len(top_lps) > 1:
            second_item = top_lps[1]
            if isinstance(second_item, (list, tuple)) and len(second_item) > 1:
                top2_lp = second_item[1]
            elif isinstance(second_item, dict) and 'logprob' in second_item:
                top2_lp = second_item['logprob']
        
        margin = top1_lp - top2_lp
        positions.append({
            "n": i,
            "token": token,
            "top1_lp": top1_lp,
            "top2_lp": top2_lp,
            "margin": margin
        })
        
    out_path = os.path.join(OUTPUT_DIR, f"margins_{label}.json")
    with open(out_path, "w") as f:
        json.dump({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "positions": positions
        }, f, indent=2)
        
    print(f"Saved {label}: {completion_tokens} tokens generated.")
    return completion_tokens

def main():
    filler = "The observer pattern is a software design pattern in which an object, named the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes. "
    
    # Approximate tokens: filler is ~30 words/tokens. 
    # 2k: ~65 repeats, 5k: ~165 repeats, 20k: ~660 repeats.
    contexts = [
        ("trivial", "Think step by step and write a very detailed 800+ token explanation of how photosynthesis works, covering chlorophyll, the light reactions, and the Calvin cycle."),
        ("ctx2k", (filler * 65) + "\n\nThink step by step and write a very detailed 800+ token explanation of how photosynthesis works, covering chlorophyll, the light reactions, and the Calvin cycle."),
        ("ctx5k", (filler * 165) + "\n\nThink step by step and write a very detailed 800+ token explanation of how photosynthesis works, covering chlorophyll, the light reactions, and the Calvin cycle."),
        ("ctx20k", (filler * 660) + "\n\nThink step by step and write a very detailed 800+ token explanation of how photosynthesis works, covering chlorophyll, the light reactions, and the Calvin cycle."),
    ]
    
    at_least_one_success = False
    for label, prompt in contexts:
        attempts = 0
        while attempts < 3:
            res = collect_margins(label, prompt)
            if res == "BLOCKER":
                print("BLOCKER encountered. Exiting.")
                return
            if res is None:
                attempts += 1
                time.sleep(10)
                continue
                
            if res >= 798:
                at_least_one_success = True
                break
            else:
                attempts += 1
                print(f"Run {label} stopped early ({res} tokens). Retrying with more open-ended prompt...")
                prompt += " Please provide significantly more detail and expand every point extensively."
        
    if not at_least_one_success:
        print("Warning: No context reached the 798 token threshold.")
    else:
        print("Margin collection finished successfully with at least one 798+ token run.")

if __name__ == "__main__":
    main()
