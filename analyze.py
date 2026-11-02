"""Turn results/raw.jsonl into results/summary.md and results/figures/*.png."""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RAW, OUT, FIG = "results/raw.jsonl", "results/summary.md", "results/figures"

# palette: fixed categorical order (validated: CVD + normal-vision separation pass)
COLOR = {"streaming": "#2a78d6", "h2o": "#eb6834", "snapkv": "#1baf7a",
         "quant": "#eda100", "snapkv+int4-kivi": "#e87ba4"}
MARK = {"streaming": "o", "h2o": "s", "snapkv": "^", "quant": "D", "snapkv+int4-kivi": "v"}
LABEL = {"streaming": "StreamingLLM", "h2o": "H2O", "snapkv": "SnapKV",
         "quant": "KIVI quant (8→4→2 bit)", "snapkv+int4-kivi": "SnapKV + INT4"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
QUANT_ORDER = ["int8", "int4-kivi", "int2-kivi"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    return rows


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows):
    """per variant: NIAH acc, LongBench avg (mean over tasks of mean score), agreement, memory."""
    by = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        suite = "niah" if r["task"] == "niah" else "lb"
        by[r["variant"]][suite].append(r)
        by[r["variant"]]["all"].append(r)
        meta[r["variant"]] = (r["method"], r["ratio"])
    out = {}
    for v, d in by.items():
        lb_tasks = defaultdict(list)
        for r in d["lb"]:
            lb_tasks[r["task"]].append(r["score"])
        out[v] = {
            "method": meta[v][0], "ratio": meta[v][1],
            "mem": mean(r["mem_frac"] for r in d["all"]),
            "niah": mean(r["score"] for r in d["niah"]),
            "lb": mean(mean(s) for s in lb_tasks.values()) if lb_tasks else float("nan"),
            "lb_tasks": {t: mean(s) for t, s in lb_tasks.items()},
            "agree": mean(r["agree_full"] for r in d["all"]),
            "k_err": mean(r["k_rel_err"] for r in d["all"] if r["k_rel_err"] is not None) if d["all"][0]["k_rel_err"] is not None else None,
            "v_err": mean(r["v_rel_err"] for r in d["all"] if r["v_rel_err"] is not None) if d["all"][0]["v_rel_err"] is not None else None,
        }
    return out


def family(method):
    return "quant" if method in ("int8", "int4-kivi", "int2-kivi", "int4-token") else method


def frontier_plot(S, key, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    full = S["full"][key]
    ax.axhline(full, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
    ax.text(1.0, full, " full fp16", color=INK2, fontsize=9, va="bottom", ha="right")
    series = defaultdict(list)
    for v, s in S.items():
        f = family(s["method"])
        if f == "full" or s["method"] == "int4-token":
            continue
        series[f].append((s["mem"], s[key], v))
    for f in ("streaming", "h2o", "snapkv", "quant", "snapkv+int4-kivi"):
        pts = sorted(series.get(f, []))
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=COLOR[f], linewidth=2, marker=MARK[f], markersize=7,
                markeredgecolor=SURF, markeredgewidth=1.5, label=LABEL[f], zorder=3)
    if "int4-token" in S:                          # the naive scheme, as a lone hollow marker
        s = S["int4-token"]
        ax.plot([s["mem"]], [s[key]], marker="D", markersize=8, markerfacecolor=SURF,
                markeredgecolor=COLOR["quant"], markeredgewidth=2, linestyle="none",
                label="INT4 per-token (naive)", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("KV-cache memory (fraction of full fp16)", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def niah_heatmaps(rows, variants, path):
    niah = [r for r in rows if r["task"] == "niah"]
    if not niah:
        return
    lengths = sorted({r["length"] for r in niah})
    depths = sorted({r["depth"] for r in niah})
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec"] + BLUES[2:])
    fig, axes = plt.subplots(1, len(variants), figsize=(2.2 * len(variants), 2.9), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, v in zip(axes, variants):
        grid = [[mean(r["score"] for r in niah if r["variant"] == v and r["length"] == L and r["depth"] == d)
                 for L in lengths] for d in depths]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(depths)):
            for j in range(len(lengths)):
                ok = grid[i][j] >= 0.5
                ax.text(j, i, "✓" if ok else "✗", ha="center", va="center", fontsize=9,
                        color="#ffffff" if ok else INK2)
        ax.set_xticks(range(len(lengths)), [f"{L // 1024}k" for L in lengths], fontsize=8, color=INK2)
        ax.set_yticks(range(len(depths)), [f"{int(d * 100)}%" for d in depths], fontsize=8, color=INK2)
        acc = mean(r["score"] for r in niah if r["variant"] == v)
        ax.set_title(f"{v}\n{acc:.0%}", fontsize=9, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("needle depth", color=INK2, fontsize=9)
    fig.supxlabel("context length (tokens)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def sparsity_plot(rows, path):
    sp = [r["sparsity"] for r in rows if r.get("sparsity")]
    if not sp:
        return
    L = len(sp[0]["per_layer"])
    top1 = [mean(s["per_layer"][l]["0.01"] for s in sp) for l in range(L)]
    top5 = [mean(s["per_layer"][l]["0.05"] for s in sp) for l in range(L)]
    sink = [mean(s["per_layer"][l]["sink"] for s in sp) for l in range(L)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    for ys, c, m, lab in ((top5, COLOR["snapkv"], "^", "top 5% of keys"),
                          (top1, COLOR["streaming"], "o", "top 1% of keys"),
                          (sink, COLOR["h2o"], "s", "first 4 tokens (sinks)")):
        ax.plot(range(L), ys, color=c, marker=m, markersize=5, linewidth=2, markeredgecolor=SURF)
        ax.text(L - 0.6, ys[-1], " " + lab, color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(-0.5, L + 5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("layer", color=INK2, fontsize=10)
    ax.set_ylabel("share of attention mass", color=INK2, fontsize=10)
    ax.set_title(f"Attention is sparse: where the question's attention goes ({len(sp)} prompts)",
                 color=INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    return {"top1": mean(top1), "top5": mean(top5), "sink": mean(sink)}


def quant_error_plot(rows, path):
    per = defaultdict(list)
    for r in rows:
        if r.get("k_rel_err_per_layer") and r["method"] in ("int8", "int4-token", "int4-kivi", "int2-kivi"):
            per[r["method"]].append(r["k_rel_err_per_layer"])
    if not per:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    spec = {"int8": (COLOR["streaming"], "o"), "int4-token": (COLOR["h2o"], "s"),
            "int4-kivi": (COLOR["snapkv"], "^"), "int2-kivi": (COLOR["quant"], "D")}
    for m in ("int2-kivi", "int4-token", "int4-kivi", "int8"):
        if m not in per:
            continue
        L = len(per[m][0])
        ys = [mean(x[l] for x in per[m]) for l in range(L)]
        c, mk = spec[m]
        ax.plot(range(L), ys, color=c, marker=mk, markersize=5, linewidth=2, markeredgecolor=SURF)
        ax.text(L - 0.6, ys[-1], " " + m, color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(-0.5, L + 4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("layer", color=INK2, fontsize=10)
    ax.set_ylabel("relative error of K  ‖K − Q(K)‖ / ‖K‖", color=INK2, fontsize=10)
    ax.set_title("Key quantization error per layer", color=INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    rows = load()
    os.makedirs(FIG, exist_ok=True)
    S = summarize(rows)
    n_niah = len({r["item"] for r in rows if r["task"] == "niah"})
    n_lb = len({r["item"] for r in rows if r["task"] != "niah"})
    frontier_plot(S, "niah", f"Needle in a haystack: accuracy vs KV memory ({n_niah} prompts, 2k–16k)",
                  "needle retrieval accuracy", f"{FIG}/frontier_niah.png")
    frontier_plot(S, "lb", f"LongBench (6 tasks, {n_lb} prompts, 4k–8k): score vs KV memory",
                  "LongBench score (mean of tasks)", f"{FIG}/frontier_longbench.png")
    niah_heatmaps(rows, ["full", "int4-kivi", "snapkv@0.1", "h2o@0.1", "streaming@0.1", "int2-kivi"],
                  f"{FIG}/niah_heatmaps.png")
    sp = sparsity_plot(rows, f"{FIG}/attention_sparsity.png")
    quant_error_plot(rows, f"{FIG}/quant_error.png")

    order = (["full", "int8", "int4-kivi", "int4-token", "int2-kivi"] +
             [f"snapkv+int4-kivi@{r}" for r in (0.5, 0.25)] +
             [f"{m}@{r}" for m in ("snapkv", "h2o", "streaming") for r in (0.5, 0.25, 0.1, 0.05)])
    tasks = sorted({t for s in S.values() for t in s["lb_tasks"]})
    lines = ["# Results", "", f"{n_niah} needle-in-a-haystack prompts (2k–16k tokens, 5 depths) and "
             f"{n_lb} LongBench-E prompts (4k–8k words, middle-truncated to 8k tokens). "
             "Qwen2.5-0.5B-Instruct, fp16, greedy. Memory = stored KV bytes of the prompt / full fp16 "
             "(quantized: payload + scales + zeros).", "",
             "| variant | KV memory | NIAH acc | LongBench | agreement w/ full | K rel. err |",
             "|---|---:|---:|---:|---:|---:|"]
    for v in order:
        if v not in S:
            continue
        s = S[v]
        ke = f"{s['k_err']:.1%}" if s["k_err"] is not None else "-"
        lines.append(f"| {v} | {s['mem']:.1%} | {s['niah']:.0%} | {100 * s['lb']:.1f} | {s['agree']:.0%} | {ke} |")
    lines += ["", "## LongBench per task (score × 100)", "",
              "| variant | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for v in order:
        if v in S:
            lines.append(f"| {v} | " + " | ".join(f"{100 * S[v]['lb_tasks'].get(t, float('nan')):.1f}" for t in tasks) + " |")
    lb_full = {t: S["full"]["lb_tasks"][t] for t in tasks}
    lines += ["", "## Needle accuracy by context length", "",
              "| variant | " + " | ".join(f"{L // 1024}k" for L in sorted({r['length'] for r in rows if r['task'] == 'niah'})) + " |",
              "|---|" + "---:|" * len({r['length'] for r in rows if r['task'] == 'niah'})]
    for v in order:
        accs = []
        for L in sorted({r['length'] for r in rows if r['task'] == 'niah'}):
            accs.append(mean(r["score"] for r in rows if r["variant"] == v and r["task"] == "niah" and r["length"] == L))
        if v in S:
            lines.append(f"| {v} | " + " | ".join(f"{a:.0%}" for a in accs) + " |")
    if sp:
        lines += ["", f"Attention sparsity (observation-window queries, mean over layers and prompts): "
                  f"top 1% of keys hold {sp['top1']:.0%} of attention mass, top 5% hold {sp['top5']:.0%}, "
                  f"the first 4 tokens hold {sp['sink']:.0%}."]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
