"""Measure outlier channels: per layer, the largest channel's max |x| over tokens divided by the
median channel's, for keys vs values (one 4k needle prompt). Run from the repo root."""
import sys, torch
sys.path.insert(0, ".")
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from kvcompress import tasks
from kvcompress.model import KVModel
d = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json","*.safetensors","*.txt"])
tok = AutoTokenizer.from_pretrained(d); m = KVModel(d)
it = tasks.niah_items(tok, lengths=(4096,), depths=(0.5,))[0]
pre = m.prefill(it["prompt_ids"])
kr, vr = [], []
for l in range(m.L):
    for name, X, acc in (("K", pre.k[l], kr), ("V", pre.v[l], vr)):
        ch = X.float().abs().amax(1)              # [nkv, hd] per-channel max |x| over tokens
        acc.append((ch.amax(-1) / ch.median(-1).values).mean().item())   # max channel / median channel
print("max/median channel magnitude ratio  K: mean %.1f  range %.1f-%.1f" % (sum(kr)/len(kr), min(kr), max(kr)))
print("                                     V: mean %.1f  range %.1f-%.1f" % (sum(vr)/len(vr), min(vr), max(vr)))
