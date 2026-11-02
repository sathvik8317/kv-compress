"""Qwen2 / Llama decoder for long-context KV-compression experiments.

Two entry points:

prefill(ids)  -> PrefillResult
    Runs the whole prompt at full precision, LAYER BY LAYER (all tokens through
    layer 0, then layer 1, ...), with attention chunked over queries so a 16k-token
    prompt fits in 4 GB. Alongside each layer's K/V it records two attention
    statistics the eviction policies need:

      h2o[l]  [nkv, S]  attention each key received, summed over EVERY query
                        of the prompt and over the query heads sharing that KV head
                        (H2O's "accumulated attention" -> heavy hitters)
      snap[l] [nkv, S]  same, but only over the last `window` queries
                        (SnapKV's "observation window", i.e. the question)

decode(variants, ...) -> generated tokens for every variant
    Each variant is a different (compressed / quantized) version of the prompt's
    KV cache. They are decoded together as one batch: at batch size 1 decode is
    launch-bound, so 19 variants cost about as much as one.

Cached keys are stored AFTER RoPE, so evicting tokens keeps their original
positions; new tokens continue at position S, S+1, ...
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass
class PrefillResult:
    k: list          # per layer [nkv, S, hd] fp16 (post-RoPE)
    v: list          # per layer [nkv, S, hd] fp16
    h2o: list        # per layer [nkv, S] fp32
    snap: list       # per layer [nkv, S] fp32
    last_logits: torch.Tensor
    seq_len: int


def _rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class KVModel:
    def __init__(self, model_dir, dtype=torch.float16, device="cuda"):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg, self.dtype, self.device = cfg, dtype, device
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.nh
        self.g = self.nh // self.nkv
        self.eps = cfg["rms_norm_eps"]

        raw = {}
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith(".safetensors"):
                raw.update(load_file(os.path.join(model_dir, fn)))
        P = lambda n: raw[n].to(device=device, dtype=dtype)
        self.embed = P("model.embed_tokens.weight")
        self.lm_head = self.embed if cfg.get("tie_word_embeddings") else P("lm_head.weight")
        self.norm = P("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            qkv_b = (torch.cat([P(a + f"{n}_proj.bias") for n in "qkv"])
                     if a + "q_proj.bias" in raw else None)
            self.layers.append(dict(
                ln1=P(p + "input_layernorm.weight"), ln2=P(p + "post_attention_layernorm.weight"),
                qkv_w=torch.cat([P(a + f"{n}_proj.weight") for n in "qkv"]), qkv_b=qkv_b,
                o_w=P(a + "o_proj.weight"),
                gu_w=torch.cat([P(p + "mlp.gate_proj.weight"), P(p + "mlp.up_proj.weight")]),
                down_w=P(p + "mlp.down_proj.weight")))
        del raw
        theta = cfg.get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    def _rope(self, x, pos):                  # x [T, heads, hd], pos [T] long
        f = torch.outer(pos.float(), self.inv_freq)
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos()[:, None].to(x.dtype), emb.sin()[:, None].to(x.dtype)
        x1, x2 = x.chunk(2, -1)
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def _qkv(self, lw, h, pos):
        T = h.shape[0]
        qkv = F.linear(h, lw["qkv_w"], lw["qkv_b"])
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
        q = self._rope(q.view(T, self.nh, self.hd), pos)
        k = self._rope(k.view(T, self.nkv, self.hd), pos)
        return q, k, v.view(T, self.nkv, self.hd)

    def _mlp(self, lw, x):
        g, u = F.linear(x, lw["gu_w"]).chunk(2, -1)
        return F.linear(F.silu(g) * u, lw["down_w"])

    # ------------------------------------------------------------------ prefill
    @torch.inference_mode()
    def prefill(self, ids, window=32, attn_budget_bytes=64 * 2**20, mlp_chunk=2048):
        S = len(ids)
        ids_t = torch.tensor(ids, device=self.device)
        pos = torch.arange(S, device=self.device)
        x = self.embed[ids_t]                                   # [S, H]
        # query chunk so the fp32 score block [nkv, g, C, S] stays under budget. Keep it SMALL:
        # variable-size chunks fragment the caching allocator, and on Windows the driver then
        # silently spills GPU memory to system RAM (256 MB budget: 10.6 GB reserved, 6x slower).
        C = max(32, min(1024, attn_budget_bytes // (self.nh * S * 4)))
        scale = self.hd ** -0.5
        res = PrefillResult([], [], [], [], None, S)
        for lw in self.layers:
            h = _rms(x, lw["ln1"], self.eps)
            q, k, v = self._qkv(lw, h, pos)
            K = k.transpose(0, 1).contiguous()                  # [nkv, S, hd]
            V = v.transpose(0, 1).contiguous()
            h2o = torch.zeros(self.nkv, S, device=self.device)
            snap = torch.zeros(self.nkv, S, device=self.device)
            attn_out = torch.empty(S, self.nh * self.hd, device=self.device, dtype=self.dtype)
            for s in range(0, S, C):
                e = min(S, s + C)
                n = e - s
                # GQA folded into the matmul: [nkv, g*n, hd] @ [nkv, hd, e]
                qg = (q[s:e] * scale).view(n, self.nkv, self.g, self.hd).permute(1, 2, 0, 3)
                scores = torch.matmul(qg.reshape(self.nkv, self.g * n, self.hd),
                                      K[:, :e].transpose(1, 2)).view(self.nkv, self.g, n, e).float()
                causal = torch.arange(e, device=self.device)[None, :] > (s + torch.arange(n, device=self.device))[:, None]
                scores.masked_fill_(causal, float("-inf"))
                probs = scores.softmax(-1)                      # [nkv, g, n, e]
                h2o[:, :e] += probs.sum((1, 2))
                r0 = max(0, S - window - s)
                if r0 < n:
                    snap[:, :e] += probs[:, :, r0:].sum((1, 2))
                o = torch.matmul(probs.to(self.dtype).view(self.nkv, self.g * n, e), V[:, :e])
                attn_out[s:e] = o.view(self.nkv, self.g, n, self.hd).permute(2, 0, 1, 3).reshape(n, -1)
                del scores, probs
            x = x + F.linear(attn_out, lw["o_w"])
            for s in range(0, S, mlp_chunk):
                xs = x[s:s + mlp_chunk]
                x[s:s + mlp_chunk] = xs + self._mlp(lw, _rms(xs, lw["ln2"], self.eps))
            res.k.append(K); res.v.append(V); res.h2o.append(h2o); res.snap.append(snap)
        last = _rms(x[-1:], self.norm, self.eps)
        res.last_logits = F.linear(last, self.lm_head).float()[0]
        return res

    # ------------------------------------------------------------------ batched decode
    @torch.inference_mode()
    def decode(self, caches, input_token, start_pos, max_new, stop_ids):
        """caches: list over variants of (k_list, v_list), each per-layer [nkv, n_i, hd] fp16,
        holding (a compressed version of) prompt tokens 0..start_pos-1.
        Every variant re-reads `input_token` (the last prompt token) at `start_pos` through
        its own cache, then decodes greedily - so every generated token depends on the
        compressed KV, none comes for free from the full-precision prefill."""
        Vn = len(caches)
        lens = torch.tensor([c[0][0].shape[1] for c in caches], device=self.device)
        cap = int(lens.max()) + max_new
        kb, vb = [], []
        for l in range(self.L):
            K = torch.zeros(Vn, self.nkv, cap, self.hd, device=self.device, dtype=self.dtype)
            Vv = torch.zeros_like(K)
            for i, (ks, vs) in enumerate(caches):
                n = ks[l].shape[1]
                K[i, :, :n] = ks[l]
                Vv[i, :, :n] = vs[l]
            kb.append(K); vb.append(Vv)
        del caches
        ar = torch.arange(Vn, device=self.device)
        kpos = torch.arange(cap, device=self.device)
        tok = torch.full((Vn,), input_token, device=self.device)
        out = [[] for _ in range(Vn)]
        done = [False] * Vn
        scale = self.hd ** -0.5
        for step in range(max_new):
            pos = torch.full((Vn,), start_pos + step, device=self.device)
            x = self.embed[tok]
            mask = kpos[None, :] <= lens[:, None]               # cached + the token written now
            for l, lw in enumerate(self.layers):
                h = _rms(x, lw["ln1"], self.eps)
                q, k, v = self._qkv(lw, h, pos)
                kb[l][ar, :, lens] = k
                vb[l][ar, :, lens] = v
                qg = (q * scale).view(Vn, self.nkv, self.g, self.hd)
                sc = torch.matmul(qg, kb[l].transpose(-1, -2)).float()   # [Vn, nkv, g, cap]
                sc.masked_fill_(~mask[:, None, None], float("-inf"))
                o = torch.matmul(sc.softmax(-1).to(self.dtype), vb[l])   # [Vn, nkv, g, hd]
                x = x + F.linear(o.reshape(Vn, -1), lw["o_w"])
                x = x + self._mlp(lw, _rms(x, lw["ln2"], self.eps))
            tok = F.linear(_rms(x, self.norm, self.eps), self.lm_head).argmax(-1)
            lens = lens + 1
            for i, t in enumerate(tok.tolist()):
                if not done[i]:
                    if t in stop_ids:
                        done[i] = True
                    else:
                        out[i].append(t)
            if all(done):
                break
        return out
