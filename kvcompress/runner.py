"""Evaluate every compression variant on one prompt.

    pre = model.prefill(prompt_ids)            # once, full precision, + attention stats
    for each variant: build its compressed cache of tokens [0, S-1)
    decode all variants in batched groups      # memory-capped
    -> answers + exact KV bytes + quantization error per variant

Variants are built lazily, one decode group at a time, so a 16k-token prompt with
19 variants still fits in 4 GB.
"""
from dataclasses import dataclass, field

import torch

from . import methods as M

RATIOS = (0.5, 0.25, 0.1, 0.05)
COMBO_RATIOS = (0.5, 0.25)


@dataclass
class Spec:
    name: str          # e.g. "snapkv@0.25"
    method: str        # full | streaming | h2o | snapkv | int8 | ... | snapkv+int4-kivi
    ratio: float       # fraction of tokens kept (1.0 for pure quantization)
    n_tokens: int      # tokens in the cache (for grouping)


def make_specs(S, ratios=RATIOS, combo_ratios=COMBO_RATIOS):
    specs = [Spec("full", "full", 1.0, S)]
    for m in ("streaming", "h2o", "snapkv"):
        for r in ratios:
            specs.append(Spec(f"{m}@{r}", m, r, max(1, round(S * r))))
    for q in M.QUANT_SCHEMES:
        specs.append(Spec(q, q, 1.0, S))
    for r in combo_ratios:
        specs.append(Spec(f"snapkv+int4-kivi@{r}", "snapkv+int4-kivi", r, max(1, round(S * r))))
    return specs


def _indices(spec, pre, l, S):
    if spec.method == "streaming":
        return M.select_streaming(S, spec.ratio, pre.k[l].shape[0], device=pre.k[l].device)
    if spec.method == "h2o":
        return M.select_h2o(pre.h2o[l][:, :S], spec.ratio)
    return M.select_snapkv(pre.snap[l][:, :S], spec.ratio)


def build(spec, pre, S):
    """Returns (k_list, v_list, info). Cache covers prompt tokens [0, S)."""
    L = len(pre.k)
    full_bytes = sum(pre.k[l][:, :S].numel() + pre.v[l][:, :S].numel() for l in range(L)) * 2
    ks, vs, stored, kerr, verr = [], [], 0, [], []
    for l in range(L):
        K, V = pre.k[l][:, :S], pre.v[l][:, :S]
        if spec.method in ("streaming", "h2o", "snapkv", "snapkv+int4-kivi"):
            idx = _indices(spec, pre, l, S)
            K, V = M.gather(K, idx), M.gather(V, idx)
        if spec.method == "full" or spec.method in ("streaming", "h2o", "snapkv"):
            stored += (K.numel() + V.numel()) * 2
        else:
            scheme = "int4-kivi" if spec.method == "snapkv+int4-kivi" else spec.method
            qk, qv = M.quantize_kv(K, V, scheme)
            stored += qk.nbytes + qv.nbytes
            dk, dv = qk.dequant(), qv.dequant()
            kerr.append(M.rel_error(K, dk)); verr.append(M.rel_error(V, dv))
            K, V = dk, dv
        ks.append(K.contiguous()); vs.append(V.contiguous())
    info = {"mem_frac": stored / full_bytes, "kv_bytes": stored}
    if kerr:
        info["k_rel_err"] = sum(kerr) / L
        info["v_rel_err"] = sum(verr) / L
        info["k_rel_err_per_layer"] = kerr
    return ks, vs, info


def groups(specs, bytes_per_token, max_new, cap_bytes=512 * 2**20):
    """Pack variants into decode batches whose PADDED buffers stay under cap_bytes."""
    out, cur, cur_max = [], [], 0
    for s in sorted(specs, key=lambda s: -s.n_tokens):
        m = max(cur_max, s.n_tokens + max_new)
        if cur and (len(cur) + 1) * m * bytes_per_token > cap_bytes:
            out.append(cur); cur, m = [], s.n_tokens + max_new
        cur.append(s); cur_max = m
    if cur:
        out.append(cur)
    return out


def run_prompt(model, ids, max_new, stop_ids, specs=None):
    """Returns {variant name: {"tokens": [...], **info}} plus attention-sparsity stats."""
    pre = model.prefill(ids)
    S = len(ids) - 1                                     # caches hold tokens [0, S)
    specs = specs or make_specs(S)
    bpt = 2 * model.L * model.nkv * model.hd * 2
    results = {}
    for grp in groups(specs, bpt, max_new):
        caches, infos = [], []
        for sp in grp:
            ks, vs, info = build(sp, pre, S)
            caches.append((ks, vs)); infos.append(info)
        outs = model.decode(caches, ids[-1], S, max_new, stop_ids)
        del caches
        for sp, info, toks in zip(grp, infos, outs):
            results[sp.name] = {"method": sp.method, "ratio": sp.ratio, "tokens": toks, **info}
        torch.cuda.empty_cache()
    results["_prefill_argmax"] = int(pre.last_logits.argmax())
    results["_sparsity"] = attention_sparsity(pre, S)
    del pre
    torch.cuda.empty_cache()
    return results


def attention_sparsity(pre, S, fracs=(0.01, 0.05, 0.1, 0.25)):
    """How concentrated is the question's attention? For the observation-window queries:
    the share of attention mass captured by the top x% of keys (per layer, averaged over
    KV heads), and the share landing on the first 4 tokens (attention sinks)."""
    per_layer = []
    for sc in pre.snap:
        p = sc[:, :S] / sc[:, :S].sum(-1, keepdim=True)
        cum = p.sort(-1, descending=True).values.cumsum(-1)
        row = {str(f): cum[:, max(1, int(S * f)) - 1].mean().item() for f in fracs}
        row["sink"] = p[:, :4].sum(-1).mean().item()
        per_layer.append(row)
    keys = list(per_layer[0])
    return {"mean": {k: sum(r[k] for r in per_layer) / len(per_layer) for k in keys},
            "per_layer": per_layer}
