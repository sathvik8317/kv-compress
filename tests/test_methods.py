"""Unit tests for selection and quantization (GPU optional: falls back to CPU)."""
import pytest
import torch

from kvcompress import methods as M

dev = "cuda" if torch.cuda.is_available() else "cpu"


def test_pack_roundtrip():
    for bits in (2, 4):
        q = torch.randint(0, 1 << bits, (2, 5, 64), dtype=torch.uint8, device=dev)
        assert torch.equal(M._unpack(M._pack(q, bits), bits), q)


@pytest.mark.parametrize("scheme", list(M.QUANT_SCHEMES))
def test_quant_shapes_and_error(scheme):
    torch.manual_seed(0)
    k = torch.randn(2, 100, 64, device=dev).half()
    k[:, :, 3] *= 20                                     # an outlier channel, like real keys
    v = torch.randn(2, 100, 64, device=dev).half()
    qk, qv = M.quantize_kv(k, v, scheme)
    dk, dv = qk.dequant(), qv.dequant()
    assert dk.shape == k.shape and dv.shape == v.shape
    bound = {"int8": 0.02, "int4-token": 0.4, "int4-kivi": 0.2, "int2-kivi": 0.8}[scheme]
    assert M.rel_error(k, dk) < bound, (scheme, M.rel_error(k, dk))
    full = (k.numel() + v.numel()) * 2
    assert qk.nbytes + qv.nbytes < full


def test_kivi_beats_per_token_on_outlier_keys():
    torch.manual_seed(0)
    k = torch.randn(2, 256, 64, device=dev).half()
    k[:, :, [5, 17]] *= 30
    tok = M.QTensor(k, 4, "token").dequant()
    chan = M.QTensor(k, 4, "channel").dequant()
    assert M.rel_error(k, chan) < M.rel_error(k, tok)


def test_selection_sizes_and_contents():
    S, nkv = 1000, 2
    score = torch.rand(nkv, S, device=dev)
    score[:, 10] = 100                                   # an obvious heavy hitter
    for r in (0.5, 0.1, 0.05):
        b = round(S * r)
        st = M.select_streaming(S, r, nkv, device=dev)
        h2 = M.select_h2o(score, r)
        sn = M.select_snapkv(score, r)
        for idx in (st, h2, sn):
            assert idx.shape == (nkv, b)
            assert (idx.sort(-1).values == idx).all()           # sorted, positions preserved
            assert all(len(set(row.tolist())) == b for row in idx)  # no duplicates
        assert (st[:, :4] == torch.arange(4, device=dev)).all()   # sinks kept
        assert (h2 == 10).any(-1).all() and (sn == 10).any(-1).all()
        assert (sn[:, -32:] == torch.arange(S - 32, S, device=dev)).all()  # observation window kept


def test_ratio_one_keeps_everything():
    S = 300
    score = torch.rand(2, S, device=dev)
    for idx in (M.select_streaming(S, 1.0, 2, device=dev), M.select_h2o(score, 1.0), M.select_snapkv(score, 1.0)):
        assert torch.equal(idx, torch.arange(S, device=dev).expand(2, -1))
