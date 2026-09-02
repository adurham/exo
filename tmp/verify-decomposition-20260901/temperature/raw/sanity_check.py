# Offline sanity check for temp_probe.py: verify (a) prompt identical to entropy_probe.py
# for --mode repetitive --fixed-words 75000 --seed 1234, (b) --temperature 1.0 lands in
# the POSTed body dicts (stubbed client, no network). Prints evidence.
import sys, json, importlib.util, types

BASE = "/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901"
sys.path.insert(0, BASE)  # not needed for import but harmless

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

orig = load("orig_probe", f"{BASE}/entropy_probe.py")
new = load("temp_probe_mod", f"{BASE}/temperature/temp_probe.py")

# (a) prompt identity
p_orig = orig.get_prompt("repetitive", 75000, 1234)
p_new = new.get_prompt("repetitive", 75000, 1234)
print("PROMPT_IDENTICAL:", p_orig == p_new, "len:", len(p_new))

# (b) stub the client and capture bodies
captured = []
class FakeClient:
    def __init__(self, *a, **k): pass
    def post_bench_chat_completions(self, body):
        captured.append(json.loads(json.dumps(body)))  # deep copy
        return {"generation_stats": {"prompt_tokens": 89408, "generation_tps": 33.0,
                "prompt_tps": 300.0, "generation_tokens": 256}}

new.ExoClient = FakeClient

argv_backup = sys.argv
sys.argv = ["temp_probe.py", "--mode", "repetitive", "--fixed-words", "75000",
            "--iterations", "2", "--warmup", "1", "--max-tokens", "8",
            "--timeout", "10", "--seed", "1234", "--temperature", "1.0",
            "--json-out", f"{BASE}/temperature/raw/sanity_body.json"]
try:
    new.main()
except SystemExit as e:
    print("main exited:", e)
finally:
    sys.argv = argv_backup

print("BODIES_CAPTURED:", len(captured))
for i, b in enumerate(captured):
    print(f"body[{i}]: temperature={b.get('temperature')!r} max_tokens={b.get('max_tokens')} "
          f"keys={sorted(b.keys())} content_len={len(b['messages'][0]['content'])}")
temps = {b.get("temperature") for b in captured}
topp = any(k in b for b in captured for k in ("top_p", "top_k", "min_p"))
print("ALL_TEMPERATURE_1.0:", temps == {1.0})
print("NO_TOP_P_TOP_K_MIN_P:", not topp)