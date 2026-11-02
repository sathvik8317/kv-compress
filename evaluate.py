"""Run every KV-compression variant on needle-in-a-haystack and LongBench.

    python evaluate.py                     # both suites -> results/raw.jsonl (resumable)
    python evaluate.py --suite niah --niah-lengths 2048 4096
    python analyze.py                      # tables + plots from results/raw.jsonl

Each line of raw.jsonl is one (prompt, variant): score, KV bytes, memory fraction,
the generated text, and quantization error where relevant.
"""
import argparse
import json
import os
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from kvcompress import tasks
from kvcompress.model import KVModel
from kvcompress.runner import run_prompt
from kvcompress.tasks import qa_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--suite", choices=["niah", "longbench", "all"], default="all")
    ap.add_argument("--niah-lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--lb-n", type=int, default=20, help="samples per LongBench task")
    ap.add_argument("--lb-max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="results/raw.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    d = snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(d)
    model = KVModel(d)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    items = []
    if a.suite in ("niah", "all"):
        items += tasks.niah_items(tok, lengths=tuple(a.niah_lengths))
    if a.suite in ("longbench", "all"):
        items += tasks.longbench_items(tok, n_per_task=a.lb_n, max_tokens=a.lb_max_tokens)

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(l)["item"] for l in f}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} prompts, {len(done)} already done, {len(todo)} to run")

    t_all = time.perf_counter()
    for n, it in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            res = run_prompt(model, it["prompt_ids"], it["max_new"], stop)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {it['id']} ({len(it['prompt_ids'])} tokens) - skipped")
            continue
        full_text = tok.decode(res["full"]["tokens"], skip_special_tokens=True)
        rows = []
        for name, r in res.items():
            if name.startswith("_"):
                continue
            text = tok.decode(r["tokens"], skip_special_tokens=True)
            rows.append({
                "item": it["id"], "task": it["task"], "prompt_tokens": len(it["prompt_ids"]),
                "length": it.get("length"), "depth": it.get("depth"),
                "variant": name, "method": r["method"], "ratio": r["ratio"],
                "mem_frac": r["mem_frac"], "kv_bytes": r["kv_bytes"],
                "score": it["score"](text),
                "agree_full": qa_f1(text, full_text) if full_text.strip() else float(text == full_text),
                "k_rel_err": r.get("k_rel_err"), "v_rel_err": r.get("v_rel_err"),
                "k_rel_err_per_layer": r.get("k_rel_err_per_layer"),
                "text": text[:300],
            })
        rows[0]["sparsity"] = res["_sparsity"]
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        full = next(r for r in rows if r["variant"] == "full")
        el = time.perf_counter() - t_all
        print(f"[{n}/{len(todo)}] {it['id']:<40} {len(it['prompt_ids']):>6} tok  "
              f"full={full['score']:.2f}  {time.perf_counter() - t0:5.1f}s  "
              f"(eta {el / n * (len(todo) - n) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
