# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
import torch.nn as nn

from heretic.modifiers.liber_moe import apply_moe_ridge, find_fused_moe


class FakeFusedExperts(nn.Module):
    def __init__(self, E=4, D=16, I=12):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, D) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(E, D, I) * 0.1)


class FakeLayer(nn.Module):
    def __init__(self, E=4, D=16, I=12):
        super().__init__()
        self.experts = FakeFusedExperts(E, D, I)


class MoeRidgeTests(unittest.TestCase):
    def _mods(self):
        E, D, I = 4, 16, 12
        layers = [nn.Module(), FakeLayer(E, D, I), FakeLayer(E, D, I)]
        return layers, E, D, I

    def test_find_fused_moe(self) -> None:
        layers, _, _, _ = self._mods()
        self.assertIsNone(find_fused_moe(layers[0]))
        self.assertIsNotNone(find_fused_moe(layers[1]))

    def test_determinism_and_norms(self) -> None:
        torch.manual_seed(0)
        layers, E, D, I = self._mods()
        r = torch.randn(D)
        r = r / r.norm()
        a = {1: torch.randn(D), 2: torch.randn(D)}
        mu_g = {1: torch.rand(E, I) + 0.1, 2: torch.rand(E, I) + 0.1}
        mu_b = {1: torch.rand(E, I) + 0.1, 2: torch.rand(E, I) + 0.1}
        before = [layers[1].experts.down_proj.data.clone(),
                  layers[2].experts.down_proj.data.clone()]
        s1 = apply_moe_ridge(layers, layers, mu_g, 10, mu_b, 10, r, a,
                             range(1, 3), 0.9, 0.5, 1.0, True)
        after_one = layers[1].experts.down_proj.data.clone()
        # Reset and re-run: bit-identical.
        layers[1].experts.down_proj.data.copy_(before[0])
        layers[2].experts.down_proj.data.copy_(before[1])
        s2 = apply_moe_ridge(layers, layers, mu_g, 10, mu_b, 10, r, a,
                             range(1, 3), 0.9, 0.5, 1.0, True)
        self.assertTrue(torch.equal(after_one,
                                    layers[1].experts.down_proj.data))
        self.assertEqual(s1, s2)
        self.assertEqual(s1["moe_layers_edited"], 2)
        # Column norms (dim=1 of [E, D, I]) preserved.
        n0 = before[0].norm(dim=1)
        n1 = layers[1].experts.down_proj.data.norm(dim=1)
        self.assertTrue(torch.allclose(n0, n1, rtol=1e-4, atol=1e-6))

    def test_empty_expert_and_empty_range(self) -> None:
        torch.manual_seed(2)
        layers, E, D, I = self._mods()
        r = torch.randn(D)
        r = r / r.norm()
        a = {1: torch.randn(D)}
        # Zero moments everywhere (experts saw nothing): no crash, no NaN.
        mu = {1: torch.zeros(E, I)}
        s = apply_moe_ridge(layers, layers, mu, 10, mu, 10, r, a,
                            range(1, 2), 0.9, 0.5, 1.0, True)
        self.assertEqual(s["moe_layers_edited"], 1)
        self.assertFalse(torch.isnan(
            layers[1].experts.down_proj.data).any())
        s = apply_moe_ridge(layers, layers, mu, 10, mu, 10, r, a,
                            range(0, 1), 0.9, 0.5, 1.0, True)
        self.assertEqual(s["moe_layers_edited"], 0)


if __name__ == "__main__":
    unittest.main()
