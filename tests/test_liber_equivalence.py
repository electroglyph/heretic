# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch

from heretic.modifiers.liber_dense import (
    optimize_column_sliced,
    ridge_remove_columns,
    select_columns,
)


def _lowrank_slice_output(X_S, W_S_base, A, B, col_norms, preserve):
    W_S = W_S_base + (B @ A)
    if preserve:
        W_S = W_S * (col_norms / W_S.norm(dim=0, keepdim=True).clamp_min(1e-8))
    return X_S @ W_S.T


class ColumnSlicedEquivalenceTests(unittest.TestCase):
    """The closure's low-rank slice math must match the dense construction."""

    def _check(self, preserve: bool) -> None:
        torch.manual_seed(0)
        d_out, d_in, r, n, k = 64, 96, 8, 24, 16
        W_base = torch.randn(d_out, d_in, dtype=torch.float64)
        X_good = torch.randn(n, d_in, dtype=torch.float64)
        X_bad = torch.randn(n, d_in, dtype=torch.float64)
        r_vec = torch.randn(d_out, dtype=torch.float64)
        r_vec = r_vec / r_vec.norm()
        S_idx = torch.arange(k)
        alpha_S = W_base[:, S_idx].T @ r_vec
        G_bad = torch.randn(n, d_out, dtype=torch.float64)
        W_new = optimize_column_sliced(
            W_base, X_good, X_bad, r_vec, G_bad, None, S_idx, alpha_S,
            gamma=0.5, lora_rank=r, w_preserve=1.0, w_kl_basis=0.0,
            w_steer=1.0, n_steps=3, seed=7,
            preserve_col_magnitudes=preserve)
        # Unselected columns are bit-identical (removal acts on bar_S only
        # via ridge_remove_columns; optimize_column_sliced never touches it).
        self.assertTrue(torch.equal(W_new[:, k:], W_base[:, k:]))
        if preserve:
            n0 = W_base[:, S_idx].norm(dim=0)
            n1 = W_new[:, S_idx].norm(dim=0)
            self.assertTrue(torch.allclose(n0, n1, rtol=1e-4, atol=1e-6))

    def test_slice_untouched_without_norm(self) -> None:
        self._check(preserve=False)

    def test_slice_norms_preserved(self) -> None:
        self._check(preserve=True)

    def test_ridge_removal_direction(self) -> None:
        torch.manual_seed(1)
        d_out, d_in, n = 32, 48, 16
        W = torch.randn(d_out, d_in, dtype=torch.float64)
        r_vec = torch.randn(d_out, dtype=torch.float64)
        r_vec = r_vec / r_vec.norm()
        Xg = torch.randn(n, d_in, dtype=torch.float64)
        Xb = torch.randn(n, d_in, dtype=torch.float64)
        S_idx, bar_idx, rho = select_columns(W, Xg, Xb, r_vec, 1.0, 0.1)
        self.assertEqual(S_idx.numel() + bar_idx.numel(), d_in)
        self.assertTrue(((rho >= 0) & (rho < 1)).all())
        alpha_all = W.T @ r_vec
        W2 = ridge_remove_columns(W, bar_idx, alpha_all, rho, r_vec, True)
        # Removal moves unselected columns away along -r (projection drops).
        before = (W[:, bar_idx].T @ r_vec)
        after = (W2[:, bar_idx].T @ r_vec)
        self.assertTrue((after.abs() <= before.abs() + 1e-9).all())
        # Selected columns untouched by ridge removal.
        self.assertTrue(torch.equal(W2[:, S_idx], W[:, S_idx]))
