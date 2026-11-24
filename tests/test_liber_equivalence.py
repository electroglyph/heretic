# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch

from heretic.modifiers.liber_dense import (
    optimize_sliced_matching,
    ridge_remove_columns,
    select_columns,
)


class SlicedMatchingTests(unittest.TestCase):
    """Structural contract of the sliced output-matching redirect."""

    def _case(self):
        torch.manual_seed(0)
        d_out, d_in, r, n, k = 64, 96, 8, 24, 16
        W_base = torch.randn(d_out, d_in, dtype=torch.float32)
        X_good = torch.randn(n, d_in, dtype=torch.float32)
        X_bad = torch.randn(n, d_in, dtype=torch.float32)
        Y_good = X_good @ W_base.T
        Y_bad = X_bad @ W_base.T
        S_idx = torch.arange(k)
        return W_base, X_good, X_bad, Y_good, Y_bad, S_idx, r

    def test_zero_steer_preserves_base(self) -> None:
        # Pure preserve (steer=0): the update must vanish; row norms exact.
        W_base, Xg, Xb, Yg, Yb, S_idx, r = self._case()
        W_new = optimize_sliced_matching(
            W_base, Xg, Xb, Yg, Yb, S_idx,
            preserve_weight=1.0, steer_weight=0.0, overcorrect_weight=1.0,
            neighbor_count=3, lora_rank=r, n_steps=5, seed=7)
        self.assertTrue(torch.allclose(W_new, W_base, atol=1e-3))

    def test_row_norms_preserved(self) -> None:
        W_base, Xg, Xb, Yg, Yb, S_idx, r = self._case()
        W_new = optimize_sliced_matching(
            W_base, Xg, Xb, Yg, Yb, S_idx,
            preserve_weight=0.8, steer_weight=0.05, overcorrect_weight=1.0,
            neighbor_count=3, lora_rank=r, n_steps=2, seed=7)
        n0 = W_base.norm(dim=1)
        n1 = W_new.norm(dim=1)
        self.assertTrue(torch.allclose(n0, n1, rtol=1e-4, atol=1e-6))

    def test_update_concentrated_on_slice(self) -> None:
        # The low-rank update lives on S columns; bar columns move only via
        # the global row-norm rescale (second-order).
        W_base, Xg, Xb, Yg, Yb, S_idx, r = self._case()
        W_new = optimize_sliced_matching(
            W_base, Xg, Xb, Yg, Yb, S_idx,
            preserve_weight=0.1, steer_weight=1.0, overcorrect_weight=1.0,
            neighbor_count=3, lora_rank=r, n_steps=2, seed=7)
        dW = (W_new - W_base).abs()
        self.assertTrue(dW[:, :16].mean() > dW[:, 16:].mean())

    def test_deterministic(self) -> None:
        W_base, Xg, Xb, Yg, Yb, S_idx, r = self._case()
        kw = dict(preserve_weight=0.8, steer_weight=0.05,
                  overcorrect_weight=1.0, neighbor_count=3, lora_rank=r,
                  n_steps=2, seed=7)
        W1 = optimize_sliced_matching(W_base, Xg, Xb, Yg, Yb, S_idx, **kw)
        W2 = optimize_sliced_matching(W_base, Xg, Xb, Yg, Yb, S_idx, **kw)
        self.assertTrue(torch.equal(W1, W2))

    def test_select_columns_partition(self) -> None:
        torch.manual_seed(1)
        d_out, d_in, n = 32, 48, 16
        W = torch.randn(d_out, d_in)
        r_vec = torch.randn(d_out)
        r_vec = r_vec / r_vec.norm()
        Xg = torch.randn(n, d_in)
        Xb = torch.randn(n, d_in)
        S_idx, bar_idx, rho = select_columns(W, Xg, Xb, r_vec, 1.0, 0.1)
        self.assertEqual(S_idx.numel() + bar_idx.numel(), d_in)
        self.assertTrue(((rho >= 0) & (rho < 1)).all())

    def test_ridge_removal_direction(self) -> None:
        torch.manual_seed(1)
        d_out, d_in, n = 32, 48, 16
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        r_vec = torch.randn(d_out, dtype=torch.float64)
        r_vec = r_vec / r_vec.norm()
        Xg = torch.randn(n, d_in, dtype=torch.float64)
        Xb = torch.randn(n, d_in, dtype=torch.float64)
        S_idx, bar_idx, rho = select_columns(W, Xg, Xb, r_vec, 1.0, 0.1)
        alpha_all = W.T @ r_vec
        W2 = ridge_remove_columns(W, bar_idx, alpha_all, rho, r_vec, True)
        # Removal moves unselected columns away along -r (projection drops).
        before = (W[:, bar_idx].T @ r_vec)
        after = (W2[:, bar_idx].T @ r_vec)
        self.assertTrue((after.abs() <= before.abs() + 1e-9).all())
        # Selected columns untouched by ridge removal.
        self.assertTrue(torch.equal(W2[:, S_idx], W[:, S_idx]))


if __name__ == "__main__":
    unittest.main()
