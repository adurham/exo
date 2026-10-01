#!/usr/bin/env python3
"""p22_prep -- build the FIXED phase-22 prompts, byte-for-byte reusable.

Run ONCE on a Mac node (needs the serving tokenizer). Every later stage
(harness runs, config sweeps, the reviewer's reproduction) loads these exact
files; nothing regenerates them, so the token stream is fixed across stages.

Source text: ~/p150_long.txt (a real long document, not synthetic filler).
Writes into the directory given by P22_OUT (default ~/p22_prompts):
  prompt_{2,8,16}k_ids.json   raw token ids, EXACTLY 2048 / 8192 / 16384 long
  prompt_{2,8,16}k_text.txt   the decoded text of those ids
  manifest.json               sha256 + sizes + token counts
"""
import hashlib
import json
import os

from tokenizers import Tokenizer

OUT = os.environ.get("P22_OUT", os.path.expanduser("~/p22_prompts"))
SRC = os.environ.get("P22_SRC", os.path.expanduser("~/p150_long.txt"))
MP = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")

os.makedirs(OUT, exist_ok=True)
tok = Tokenizer.from_file(MP + "/tokenizer.json")
ids_all = tok.encode(open(SRC).read(), add_special_tokens=False).ids

manifest = {
    "source": SRC,
    "tokenizer": MP + "/tokenizer.json",
    "corpus_tokens": len(ids_all),
    "corpus_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
    "prompts": {},
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


for L in (2048, 8192, 16384):
    tag = f"{L // 1024}k"
    ids = [int(t) for t in ids_all[:L]]
    text = tok.decode(ids)
    # exact round trip is what makes "byte-for-byte reusable" true
    assert tok.encode(text, add_special_tokens=False).ids == ids, f"{tag} roundtrip"
    ids_bytes = json.dumps(ids).encode()
    text_bytes = text.encode()
    open(os.path.join(OUT, f"prompt_{tag}_ids.json"), "wb").write(ids_bytes)
    open(os.path.join(OUT, f"prompt_{tag}_text.txt"), "wb").write(text_bytes)
    manifest["prompts"][tag] = {
        "tokens": L,
        "ids_sha256": sha(ids_bytes),
        "ids_bytes": len(ids_bytes),
        "text_sha256": sha(text_bytes),
        "text_bytes": len(text_bytes),
    }

open(os.path.join(OUT, "manifest.json"), "w").write(json.dumps(manifest, indent=1))
print(json.dumps(manifest, indent=1), flush=True)
print("P22_PREP_DONE", flush=True)
