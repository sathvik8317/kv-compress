"""Why does H2O lose the needle while SnapKV keeps it? Rank the needle's tokens under each score.

For one needle prompt, reports per layer the percentile rank of the needle tokens
(100 = most-attended key) under H2O's accumulated attention vs SnapKV's
observation-window attention, and how many layers would keep each needle token
at a 10% budget. Writes results/needle_ranks.json and results/figures/needle_ranks.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from analyze import COLOR, INK, INK2, MARK, SURF, style
from kvcompress import tasks
from kvcompress.model import KVModel


def main(length=4096, depth=0.5):
    d = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(d)
    model = KVModel(d)
    it = next(i for i in tasks.niah_items(tok, lengths=(length,), depths=(depth,)))
    ids = it["prompt_ids"]
    num_ids = tok.encode(it["answer"])
    # locate the number inside the prompt (the needle's payload tokens)
    pos = next(p for p in range(len(ids) - len(num_ids)) if ids[p:p + len(num_ids)] == num_ids)
    needle = list(range(pos, pos + len(num_ids)))
    pre = model.prefill(ids)
    S = len(ids) - 1
    out = {"h2o": [], "snapkv": []}
    for l in range(model.L):
        for name, sc in (("h2o", pre.h2o[l][:, :S]), ("snapkv", pre.snap[l][:, :S])):
            ranks = sc.argsort(-1).argsort(-1).float() / (S - 1) * 100      # percentile, per head
            out[name].append(ranks[:, needle].mean().item())
    keep10 = {}
    for name, scores in (("h2o", pre.h2o), ("snapkv", pre.snap)):
        k = int(0.1 * S)
        kept = sum(int(torch.isin(torch.tensor(needle, device=s.device), s[:, :S].topk(k, -1).indices).float().mean().item() > 0.5)
                   for s in scores)
        keep10[name] = kept
    res = {"prompt": it["id"], "needle_tokens": len(needle), "needle_position": pos, "S": S,
           "mean_percentile": {k: sum(v) / len(v) for k, v in out.items()},
           "per_layer_percentile": out, "layers_keeping_needle_top10pct": keep10}
    json.dump(res, open("results/needle_ranks.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_layer_percentile"}, indent=2))

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    for name, lab in (("snapkv", "SnapKV score (question's attention)"), ("h2o", "H2O score (accumulated attention)")):
        ys = out[name]
        ax.plot(range(model.L), ys, color=COLOR[name], marker=MARK[name], markersize=5, linewidth=2,
                markeredgecolor=SURF)
        ax.text(model.L - 0.6, ys[-1], " " + lab.split(" (")[0], color=INK2, fontsize=8.5, va="center")
    ax.axhline(90, color="#898781", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0, 90.5, "kept at a 10% budget above this line", color=INK2, fontsize=8, va="bottom")
    ax.set_xlim(-0.5, model.L + 4)
    ax.set_ylim(0, 102)
    ax.set_xlabel("layer", color=INK2, fontsize=10)
    ax.set_ylabel("needle tokens' rank (percentile)", color=INK2, fontsize=10)
    ax.set_title(f"Where the needle ranks among {S:,} keys ({length // 1024}k prompt, depth {int(depth * 100)}%)",
                 color=INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig("results/figures/needle_ranks.png", facecolor=SURF)


if __name__ == "__main__":
    main()
