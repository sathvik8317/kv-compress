"""KV-cache compression strategies.

Two families:

EVICTION - keep only `budget` of the S prompt tokens, chosen per layer and per KV head
  streaming  StreamingLLM (Xiao et al. 2023): first `sink` tokens + the most recent ones.
             Ignores content entirely.
  h2o        Heavy-Hitter Oracle (Zhang et al. 2023): keys that received the most
             accumulated attention over all prompt queries, plus a recent window.
  snapkv     SnapKV (Li et al. 2024): keys the last `window` queries (the question)
             attend to, smoothed with 1-D max-pooling so neighbours of a hit survive
             too, plus the observation window itself.

QUANTIZATION - keep every token, store fewer bits per number
  int8       symmetric, one fp16 scale per (token, head)
  int4/int2  asymmetric (min/max), groups of 32 values, fp16 scale + zero per group
             "token"  groups run along head_dim (each token quantized on its own)
             "kivi"   KIVI (Liu et al. 2024): KEYS grouped per CHANNEL along the token
                      axis, VALUES per token. Keys have a few outlier channels with
                      huge magnitude; per-token groups let them blow up the scale of
                      every other channel in the group.

Quantized tensors are really packed (2 x int4 or 4 x int2 per byte), and memory is
measured from the stored tensors (payload + scales + zeros), not estimated.
"""
import math

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- eviction
def _budget(S, ratio, min_keep):
    return max(min_keep, min(S, int(round(S * ratio))))


def select_streaming(S, ratio, nkv, sink=4, device="cuda"):
    b = _budget(S, ratio, sink + 1)
    idx = torch.cat([torch.arange(sink, device=device), torch.arange(S - (b - sink), S, device=device)])
    return idx.unique().expand(nkv, -1)


def select_h2o(score, ratio, recent_frac=0.5):
    """score [nkv, S] accumulated attention. Half the budget = heavy hitters, half = recent."""
    nkv, S = score.shape
    b = _budget(S, ratio, 2)
    r = max(1, int(b * recent_frac))
    hh = b - r
    recent = torch.arange(S - r, S, device=score.device).expand(nkv, -1)
    if hh == 0:
        return recent
    heavy = score[:, :S - r].topk(hh, dim=-1).indices
    return torch.cat([heavy, recent], -1).sort(-1).values


def select_snapkv(score, ratio, window=32, kernel=7):
    """score [nkv, S] attention from the observation window (last `window` queries)."""
    nkv, S = score.shape
    b = _budget(S, ratio, window + 1)
    w = min(window, b)
    win = torch.arange(S - w, S, device=score.device).expand(nkv, -1)
    k = b - w
    if k == 0:
        return win
    prefix = score[:, :S - w]
    pooled = F.max_pool1d(prefix[None], kernel, stride=1, padding=kernel // 2)[0]
    top = pooled.topk(min(k, S - w), dim=-1).indices
    return torch.cat([top, win], -1).sort(-1).values


def gather(x, idx):
    """x [nkv, S, hd], idx [nkv, b] -> [nkv, b, hd]"""
    return x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))


# ----------------------------------------------------------------------------- quantization
def _pack(q, bits):
    """q uint8 values in [0, 2^bits) -> packed bytes along the last dim."""
    per = 8 // bits
    q = q.view(*q.shape[:-1], -1, per)
    out = torch.zeros(q.shape[:-1], dtype=torch.uint8, device=q.device)
    for i in range(per):
        out |= q[..., i] << (bits * i)
    return out


def _unpack(p, bits):
    per = 8 // bits
    mask = (1 << bits) - 1
    return torch.stack([(p >> (bits * i)) & mask for i in range(per)], -1).flatten(-2)


class QTensor:
    """A quantized [nkv, S, hd] tensor. `nbytes` counts everything that is stored."""

    def __init__(self, x, bits, axis, group=32):
        self.shape, self.dtype, self.bits, self.axis, self.group = x.shape, x.dtype, bits, axis, group
        nkv, S, hd = x.shape
        self.tail = None
        if bits == 8:                                   # symmetric per (token, head)
            scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
            self.q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale, self.zero = scale.half(), None
            return
        if axis == "channel":                           # groups of `group` TOKENS per channel
            full = (S // group) * group
            if full < S:                                # leftover tokens stay fp16 (KIVI's residual)
                self.tail = x[:, full:].clone()
            if full == 0:                               # fewer tokens than one group: all fp16
                self.q = torch.zeros(0, dtype=torch.uint8, device=x.device)
                self.scale = torch.zeros(0, dtype=torch.float16, device=x.device)
                self.zero = None
                return
            xs = x[:, :full].float().view(nkv, full // group, group, hd)
            red = 2
        else:                                           # groups of `group` CHANNELS per token
            xs = x.float().view(nkv, S, hd // group, group)
            red = 3
        lo, hi = xs.amin(red, keepdim=True), xs.amax(red, keepdim=True)
        levels = (1 << bits) - 1
        scale = ((hi - lo) / levels).clamp_min(1e-6)
        q = ((xs - lo) / scale).round().clamp(0, levels).to(torch.uint8)
        if axis == "channel":                           # pack along tokens-within-group
            q = q.transpose(2, 3)                       # [nkv, ngroups, hd, group]
        self.q = _pack(q.contiguous(), bits)
        self.scale, self.zero = scale.half(), lo.half()

    @property
    def nbytes(self):
        n = self.q.numel() * self.q.element_size() + self.scale.numel() * 2
        if self.zero is not None:
            n += self.zero.numel() * 2
        if self.tail is not None:
            n += self.tail.numel() * 2
        return n

    def dequant(self):
        nkv, S, hd = self.shape
        if self.bits == 8:
            return (self.q.float() * self.scale.float()).to(self.dtype)
        if self.q.numel() == 0:
            return self.tail
        q = _unpack(self.q, self.bits).float()
        if self.axis == "channel":
            q = q.transpose(2, 3)                       # back to [nkv, ngroups, group, hd]
        x = q * self.scale.float() + self.zero.float()
        if self.axis == "channel":
            x = x.reshape(nkv, -1, hd)
            if self.tail is not None:
                x = torch.cat([x, self.tail.float()], 1)
        else:
            x = x.reshape(nkv, S, hd)
        return x.to(self.dtype)


QUANT_SCHEMES = {
    # name: (key bits, key axis, value bits, value axis)
    "int8":        (8, "token", 8, "token"),
    "int4-token":  (4, "token", 4, "token"),
    "int4-kivi":   (4, "channel", 4, "token"),
    "int2-kivi":   (2, "channel", 2, "token"),
}


def quantize_kv(k, v, scheme):
    kb, ka, vb, va = QUANT_SCHEMES[scheme]
    return QTensor(k, kb, ka), QTensor(v, vb, va)


def rel_error(x, y):
    return ((x.float() - y.float()).norm() / x.float().norm().clamp_min(1e-9)).item()
