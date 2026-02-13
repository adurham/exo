import urllib.request
import urllib.error
import urllib.parse
import json
import time
import sys

# Configuration
MASTER_URL = "http://192.168.86.202:52415"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')

def get_json(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')

def place_instance():
    print(f"Requesting placement for {MODEL_ID} with MlxJaccl (RDMA)...")
    payload = {
        "model_id": MODEL_ID,
        "sharding": "Pipeline",
        "instance_meta": "MlxJaccl",
        "min_nodes": 2
    }
    status, resp = post_json(f"{MASTER_URL}/place_instance", payload)
    print(f"Place response: {status} {json.dumps(resp)}")
    return status == 200

def check_inference():
    print("Waiting 10s for instance to initialize...")
    time.sleep(10)
    print("Attempting inference...")
    try:
        status, resp = post_json(f"{MASTER_URL}/v1/chat/completions", {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        })
        print(f"Inference response: {status} {resp}")
        if status == 200:
            print("SUCCESS: Inference worked!")
            return True
        else:
            print("FAILURE: Inference failed")
            return False
    except Exception as e:
        print(f"FAILURE: Inference request error: {e}")
        return False

def check_instance_state():
    print("Waiting for instance to appear in state and be usable...")
    start_time = time.time()
    while time.time() - start_time < 90:  # Wait up to 90 seconds
        status, state = get_json(f"{MASTER_URL}/state")
        if status == 200:
            instances = state.get("instances", {})
            found_candidate = False
            for iid, inst in instances.items():
                if "MlxJacclInstance" in inst:
                    print(f"DEBUG: Found MlxJaccl instance {iid}, checking inference...")
                    return check_inference()
                
                # Also check inside unwrapped dict just in case
                for key in inst:
                    if isinstance(inst[key], dict) and "jacclDevices" in inst[key]:
                        print(f"DEBUG: Found MlxJaccl instance {iid} (nested), checking inference...")
                        return check_inference()
        else:
             print(f"State check failed: {status}")
        time.sleep(2)
    print("FAILURE: MlxJaccl instance did not appear or initialize in time.")
    return False

if __name__ == "__main__":
    if place_instance():
        success = check_instance_state()
        sys.exit(0 if success else 1)
    else:
        print("Placement failed")
        sys.exit(1)
