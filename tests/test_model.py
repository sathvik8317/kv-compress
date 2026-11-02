"""End-to-end correctness on the real model (needs a GPU; downloads Qwen2.5-0.5B once)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def env():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from kvcompress.model import KVModel
    d = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    return d, AutoTokenizer.from_pretrained(d), KVModel(d)


def _prompt(tok):
    text = ("Here is a short story. " + "The quick brown fox jumps over the lazy dog. " * 40 +
            "The fox's name is Rusty. What is the fox's name?")
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True))


def test_full_cache_matches_hf_generate(env):
    d, tok, model = env
    from transformers import AutoModelForCausalLM
    from kvcompress.runner import Spec, run_prompt
    ids = _prompt(tok)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    res = run_prompt(model, ids, 24, stop, specs=[Spec("full", "full", 1.0, len(ids) - 1)])
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16).cuda().eval()
    ref = hf.generate(torch.tensor([ids]).cuda(), max_new_tokens=24, do_sample=False,
                      repetition_penalty=1.0, eos_token_id=list(stop))[0, len(ids):].tolist()
    del hf; torch.cuda.empty_cache()
    ref = [t for t in ref if t not in stop]
    assert res["full"]["tokens"][:16] == ref[:16], (tok.decode(res["full"]["tokens"]), tok.decode(ref))
    # re-reading the last prompt token through the cache must agree with the prefill's own logits
    assert res["full"]["tokens"][0] == res["_prefill_argmax"]


def test_ratio_one_equals_full_and_h2o_sums(env):
    _, tok, model = env
    from kvcompress.runner import Spec, run_prompt
    ids = _prompt(tok)
    S = len(ids) - 1
    specs = [Spec("full", "full", 1.0, S)] + [Spec(m, m, 1.0, S) for m in ("streaming", "h2o", "snapkv")]
    res = run_prompt(model, ids, 16, {tok.eos_token_id}, specs=specs)
    for m in ("streaming", "h2o", "snapkv"):
        assert res[m]["tokens"] == res["full"]["tokens"], m
        assert abs(res[m]["mem_frac"] - 1.0) < 1e-9
    pre = model.prefill(ids)
    # every softmax row sums to 1: summed over all S queries and g heads per KV head -> S*g
    for h in pre.h2o:
        assert torch.allclose(h.sum(-1), torch.full_like(h.sum(-1), len(ids) * model.g), rtol=1e-3)
    for s in pre.snap:
        assert torch.allclose(s.sum(-1), torch.full_like(s.sum(-1), 32 * model.g), rtol=1e-3)
