import sys
import time
import requests
import json

BASE_URL = "http://localhost:52415"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/models")
            if resp.status_code == 200:
                print("Server is up!")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    print("Server failed to start")
    sys.exit(1)

def run_test():
    # 0. Cleanup existing instances
    print("Checking for existing instances to clean up...")
    resp = requests.get(f"{BASE_URL}/state")
    resp.raise_for_status()
    state = resp.json()
    for inst_id in state.get("instances", {}).keys():
        print(f"Deleting existing instance: {inst_id}")
        del_resp = requests.delete(f"{BASE_URL}/instance/{inst_id}")
        if not del_resp.ok:
            print(f"Failed to delete {inst_id}: {del_resp.text}")
    print("Cleanup complete.")
    time.sleep(2) # Give it a moment to clear
    
    # 1. Get Previews
    print("Getting instance previews...")
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    resp = requests.get(f"{BASE_URL}/instance/previews", params={"model_id": model_id})
    resp.raise_for_status()
    previews = resp.json()["previews"]
    if not previews:
        print(f"No previews found for {model_id}")
        sys.exit(1)
    
    first_preview = previews[0]

    # 2. Create Instance
    print(f"Creating instance for {model_id}...")
    create_req = {
        "model_id": first_preview["model_id"],
        "instance": first_preview["instance"]
    }
    
    resp = requests.post(f"{BASE_URL}/instance", json=create_req)
    resp.raise_for_status()
    print("Instance creation request sent.")

    # Wait for the instance to transition to ready
    print("Waiting for instance to be ready...")
    ready = False
    for _ in range(120):
        resp = requests.get(f"{BASE_URL}/state")
        resp.raise_for_status()
        state = resp.json()
        instances = state.get("instances", {})
        
        for inst_id, inst_wrap in instances.items():
            if not inst_wrap:
                continue
            # The value inside {"MlxRingInstance": {...}}
            instance = list(inst_wrap.values())[0] if isinstance(inst_wrap, dict) else inst_wrap
            
            shard_assignments = instance.get("shardAssignments", {})
            if getattr(shard_assignments, "get", lambda x, y=None: None)("modelId") == model_id or instance.get("model_id") == model_id:
                if instance.get("hostsByNode") or instance.get("status") == "ready":
                    ready = True
                    print(f"Instance {inst_id} is ready!")
                    break
        if ready:
            break
        time.sleep(1)

    if not ready:
        print("Instance failed to become ready")
        sys.exit(1)

    # 3. Prompt
    print(f"Sending prompt to model: {model_id}")
    chat_req = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.0
    }
    
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=chat_req)
    if resp.status_code != 200:
        print(f"Chat completion failed: {resp.text}")
        sys.exit(1)
        
    completion = resp.json()
    response_text = completion["choices"][0]["message"]["content"]
    print(f"\nResponse: {response_text}\n")
    
    if "Paris" not in response_text:
        print("Test Failed: Expected 'Paris' in response")
        sys.exit(1)
        
    print("Smoke test passed successfully!")

if __name__ == "__main__":
    wait_for_server()
    run_test()
