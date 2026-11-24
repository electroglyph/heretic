# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
import torch.nn as nn

from heretic.modifiers import liber_conv as lc


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class StubTok:
    def __init__(self, ids):
        self._ids = ids

    def __call__(self, texts, **kwargs):
        n = len(texts)
        return Batch({"input_ids": self._ids[:n].clone(),
                      "attention_mask": torch.ones_like(self._ids[:n])})


class ToyPostConv(nn.Module):
    """Linear writer -> depthwise causal conv -> residual add."""

    def __init__(self, d=16, k=4, bias=False):
        super().__init__()
        torch.manual_seed(7)
        self.embed = nn.Embedding(64, 32)
        self.writer = nn.Linear(32, d, bias=False)
        self.post = nn.Conv1d(d, d, kernel_size=k, groups=d, bias=bias,
                              padding=k - 1)
        # causal: trim future padding
        self._k = k

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed(input_ids).float()
        y = self.writer(x)
        z = self.post(y.transpose(1, 2))[:, :, : y.shape[1]].transpose(1, 2)
        # residual add against the first 16 input dims
        return z + x[:, :, :16]


class ConvScanTests(unittest.TestCase):
    def _model(self, **kw):
        torch.manual_seed(0)
        m = ToyPostConv(**kw)
        m.eval()
        return m

    def test_detect_and_fix(self) -> None:
        m = self._model()
        r = torch.randn(16)
        r = r / r.norm()
        tok = StubTok(torch.randint(0, 64, (4, 12)))
        writers = {0: [m.writer]}
        specs, post_mods = lc.scan_post_ops(m, [m], writers, tok, ["a", "b", "c", "d"])
        self.assertEqual(specs[0][0]["kind"], "depthwise_conv")
        self.assertEqual(len(specs[0][0]["taps"]), 4)
        # functional probe: plain s=1 removal still leaks through the conv
        ratio_plain = lc.probe_leak(m, m.writer, m.post, r, tok, ["a", "b"])
        self.assertGreater(ratio_plain, 0.10)
        # S-span fix kills it (verified by direct post-output projection;
        # the probe ratio is meaningless once S is removed, since plain
        # r-removal still leaks by design and the probe divides by ~0)
        S = lc.build_leak_span(r, [specs[0][0]])
        self.assertGreaterEqual(S.shape[1], 2)  # Q + at least one tap copy
        W = m.writer.weight.detach().float()
        fixed = W - S @ (S.T @ W)
        m.writer.weight.data.copy_(fixed)
        self.assertLess(
            lc.post_output_projection(m, m.post, r, tok, ["a", "b"]), 1e-5)

    def test_clean_model_no_false_positive(self) -> None:
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(3)
                self.embed = nn.Embedding(64, 32)
                self.writer = nn.Linear(32, 16, bias=False)

            def forward(self, input_ids, attention_mask=None,
                        use_cache=False):
                x = self.embed(input_ids).float()
                return self.writer(x) + x[:, :, :16]

        torch.manual_seed(1)
        m = Plain()
        m.eval()
        tok = StubTok(torch.randint(0, 64, (4, 12)))
        specs, _ = lc.scan_post_ops(m, [m], {0: [m.writer]}, tok, ["a", "b"])
        self.assertEqual(specs[0][0]["kind"], "clean")

    def test_registry_roundtrip(self) -> None:
        import tempfile, os
        tmp = os.path.join(tempfile.mkdtemp(), "reg.json")
        orig = lc.REGISTRY_PATH
        lc.REGISTRY_PATH = tmp
        try:
            self.assertEqual(lc.load_registry(), {})
            lc.save_registry({"fp1": {"verdict": "clean"}})
            self.assertEqual(lc.load_registry()["fp1"]["verdict"], "clean")
        finally:
            lc.REGISTRY_PATH = orig

    def test_fingerprint_stable(self) -> None:
        m1 = self._model()
        m2 = self._model()
        layers = [m1]
        self.assertEqual(lc.fingerprint_model(m1, layers),
                         lc.fingerprint_model(m2, layers))


if __name__ == "__main__":
    unittest.main()
