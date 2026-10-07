#!/usr/bin/env python3
"""Phase-19 agentic replay campaign driver (single process, per-rep idle-guard).

Two arms, same process, blocks (so each arm's prefix cache stays resident):
  benign  : synthetic 'count' filler @100K   (control, reproduces 28.08 baseline shape)
  agentic : real session 20261007_092009_9a2ed7 content @~89K

Idle-guard before EVERY rep:
  (a) state.db MAX(ended_at) provider='custom'  -> must be >10 min old
      (my own direct-HTTP reps never write state.db, so this only sees foreign traffic)
  (b) cluster exo.log last 'POST /v1/chat/completions' -> must be >10 min old,
      EXCLUDING my own requests (matched by response 'created' epoch +/- 3 s,
      verified to equal the server log's local timestamp to the second).
Foreign traffic mid-run -> abort that arm (never fabricate).

NO relaunch. Read-only cluster access except the inference requests themselves.
"""
import datetime, importlib.util, json, os, re, sqlite3, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)

_spec = importlib.util.spec_from_file_location("harness", os.path.join(HERE, "phase19_agentic_measure.py"))
M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M)

SSH = ["ssh", "macstudio-m4-1",
       "grep -a 'POST /v1/chat/completions' ~/.exo/exo_log/exo.log | tail -40"]
LOGRE = re.compile(r"\[ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \|")
GAMMA = 3
MINE_TOL = 5.0


def log_posts():
    r = subprocess.run(SSH, capture_output=True, text=True)
    out = (r.stdout or r.stderr or "").strip().splitlines()
    posts = []
    for line in out:
        mm = LOGRE.search(line)
        if mm:
            ep = datetime.datetime.strptime(mm.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            posts.append((ep, line.strip()))
    return posts


def guard(tag, mine):
    now = time.time()
    con = sqlite3.connect("file:/Users/adam.durham/.hermes/state.db?mode=ro", uri=True)
    m = float(con.execute("SELECT MAX(ended_at) FROM api_calls WHERE provider='custom'").fetchone()[0])
    con.close()
    db_age = now - m
    posts = log_posts()
    foreign = [(ep, ln) for ep, ln in posts if not any(abs(ep - me) <= MINE_TOL for me in mine)]
    newest_foreign = max((ep for ep, _ in foreign), default=None)
    foreign_age = (now - newest_foreign) if newest_foreign is not None else None
    # log check passes if the window holds no foreign posts (log rotated / window
    # saturated by our own requests); the state.db check still catches foreign traffic.
    ok = (db_age > 600) and (foreign_age is None or foreign_age > 600)
    rec = {"tag": tag, "now_local": datetime.datetime.fromtimestamp(now).isoformat(),
           "last_custom_call_local": datetime.datetime.fromtimestamp(m).isoformat(),
           "db_age_s": round(db_age, 1),
           "last_foreign_cluster_post_line": next((ln for ep, ln in foreign if ep == newest_foreign), None),
           "last_foreign_cluster_post_local": (datetime.datetime.fromtimestamp(newest_foreign).isoformat()
                                               if newest_foreign is not None else None),
           "foreign_age_s": round(foreign_age, 1) if foreign_age is not None else None,
           "excluded_own_requests": sorted(mine), "ok": ok}
    print("GUARD", json.dumps(rec), flush=True)
    return ok, rec


def do_arm(name, prompt, meta, reps, max_tokens, mine, guards):
    print(f"=== ARM {name}: {json.dumps(meta)} ===", flush=True)
    recs = []; prev_cyc = prev_acc = None
    for i in range(reps):
        ok, rec = guard(f"{name}_rep{i}", mine); guards.append(rec)
        if not ok:
            print(f"ABORT: foreign traffic before {name} rep {i}; discarding arm.", flush=True)
            return recs, False
        r = M.stream_once(prompt, max_tokens, None)
        if r.get("request_send_epoch"):
            mine.add(float(r["request_send_epoch"]))
            json.dump(sorted(mine), open(os.path.join(OUT, "own_requests.json"), "w"))
        r.update(M.derive(r, prev_cyc, prev_acc, GAMMA)); r["rep"] = i; r["arm"] = name
        recs.append(r)
        if r.get("cycles_cum") is not None:
            prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
        print(json.dumps({k: r.get(k) for k in (
            "arm", "rep", "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "content_chars", "reasoning_chars", "ttft_s", "decode_s", "decode_tps",
            "rounds", "mean_accepted", "ms_per_round", "gamma_implied",
            "prefix_hit", "prompt_tps", "finish_reason", "wall_s")}), flush=True)
        time.sleep(2)
    return recs, True


def summarize(name, recs, meta):
    timed = recs[1:] if len(recs) > 1 else recs
    def col(k):
        return [r[k] for r in timed if r.get(k) is not None]
    return {"arm": name, "meta": meta, "n_reps": len(recs),
            "cold_rep": ({k: recs[0].get(k) for k in ("prompt_tokens", "ttft_s", "prompt_tps",
                                                      "decode_tps", "wall_s", "prefix_hit")} if recs else None),
            "timed_reps": len(timed),
            "decode_tps_all": col("decode_tps"),
            "decode_tps_median": round(statistics.median(col("decode_tps")), 3) if col("decode_tps") else None,
            "mean_accepted_all": col("mean_accepted"),
            "mean_accepted_median": round(statistics.median(col("mean_accepted")), 4) if col("mean_accepted") else None,
            "ms_per_round_all": col("ms_per_round"),
            "ms_per_round_median": round(statistics.median(col("ms_per_round")), 2) if col("ms_per_round") else None,
            "rounds_all": col("rounds"),
            "gamma_implied_all": col("gamma_implied"),
            "reasoning_tokens_all": col("reasoning_tokens"),
            "completion_tokens_all": col("completion_tokens"),
            "prefix_hit_all": col("prefix_hit")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="both", choices=["both", "agentic", "benign"])
    a = ap.parse_args()
    only = a.only
    salt_b = "b1a2c3d4"   # fixed so retries reuse the server-side prefix cache
    prompt_b = M.build_benign_prompt(100000, salt_b, "count")
    meta_b = {"prompt_chars": len(prompt_b), "pred_tokens_3p48": int(len(prompt_b) / 3.479),
              "depth_target": 100000, "task": "count", "salt": salt_b}
    salt_a = "e5f6a7b8"   # fixed, same reason
    prompt_a, meta_a = M.build_agentic_prompt(salt_a, None)
    meta_a["salt"] = salt_a
    print("PROMPTS", json.dumps({"benign": meta_b, "agentic": meta_a}), flush=True)

    mine = set(json.load(open(os.path.join(OUT, "own_requests.json")))) \
        if os.path.exists(os.path.join(OUT, "own_requests.json")) else set()
    guards = []
    ok, first = guard("before_campaign", mine); guards.append(first)
    if not ok:
        print("GUARD FAILED before campaign; aborting.", flush=True)
        json.dump({"guards": guards}, open(os.path.join(OUT, "guards.json"), "w"), indent=1)
        return 3

    b_recs, b_ok = ([], True) if only == "agentic" else do_arm("benign", prompt_b, meta_b, 4, 1000, mine, guards)
    a_recs, a_ok = ([], True) if only == "benign" else do_arm("agentic", prompt_a, meta_a, 4, 1000, mine, guards)

    result = {"session_replayed": M.SESSION_ID, "model": M.MODEL, "gamma": GAMMA,
              "benign_complete": b_ok, "agentic_complete": a_ok,
              "benign": {"summary": summarize("benign", b_recs, meta_b), "reps": b_recs},
              "agentic": {"summary": summarize("agentic", a_recs, meta_a), "reps": a_recs},
              "guards": guards}
    json.dump(result, open(os.path.join(OUT, "campaign.json"), "w"), indent=1)
    json.dump(guards, open(os.path.join(OUT, "guards.json"), "w"), indent=1)
    print("CAMPAIGN DONE benign_complete=%s agentic_complete=%s" % (b_ok, a_ok), flush=True)
    print("BENIGN_SUMMARY", json.dumps(result["benign"]["summary"]), flush=True)
    print("AGENTIC_SUMMARY", json.dumps(result["agentic"]["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
