# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA column-sliced multi-rank LoRA for 2D writer matrices (experimental).
#
# Sliced output matching on write atoms (columns): only the top-(1-Q)
# refusal-active columns S are optimized, toward the good-output distribution
# (kNN toward good, overcorrected away from bad, ARA-style), with isotropic
# preserve on good inputs and row-norm rescaling. Unselected columns stay
# bit-identical. Reference: heretic2 §2.1–2.3, §3.3.
#
# What did NOT survive validation (local 3090, Spark-X2.6-4B, Nov 2026):
# - Analytic -r shift target (optimize_column_sliced, removed): LBFGS stalled
#   after 1 eval from N(0, 0.01) init; even kaiming + outer steps moved weights
#   without moving refusals (0/12 flips at max steer). Formulation inert.
# - Closed-form ridge removal on bar columns: inert alone, adds nothing on top
#   of sliced matching. Dropped from the dense path (kept for the leaky path
#   via apply_s_ridge).
# - Harmlessness-basis (U_H) KL penalty: zero effect at rank 7 and rank 64,
#   w_kl 5 and 20, with and without row-norm. Dropped from the dense path.

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
from torch.optim import LBFGS


def _knn_mean_distances(a: Tensor, b: Tensor, k: int) -> Tensor:
    """Mean distance of each row of a to its k nearest rows of b."""
    k = max(1, min(k, b.shape[0]))
    distances = torch.cdist(a, b)
    nearest, _ = distances.topk(k, dim=1, largest=False)
    return nearest.mean(dim=1)


def optimize_sliced_matching(
    W_base: Tensor,
    X_good: Tensor,
    X_bad: Tensor,
    Y_good: Tensor,
    Y_bad: Tensor,
    S_idx: Tensor,
    preserve_weight: float,
    steer_weight: float,
    overcorrect_weight: float,
    neighbor_count: int,
    lora_rank: int,
    n_steps: int,
    seed: int = 42,
) -> Tensor:
    """Sliced output-matching redirect on 2D writer matrices.

    Optimizes a low-rank update to the S columns so bad-input outputs move
    toward the good-output distribution (and away from bad outputs,
    overcorrected), while good-input outputs are preserved. Row norms are
    rescaled to the base norms (validated: halves first-token KL vs no norm).

    Returns the full updated weight (only S columns changed).
    """
    device = W_base.device
    k = S_idx.numel()
    eff_rank = max(1, min(lora_rank, k, W_base.shape[0]))
    gen = torch.Generator(device=device).manual_seed(seed)
    # Kaiming init (NOT near-zero): with B=0, near-zero A stalls strong-Wolfe
    # line search at the init point (1 eval, zero update). ARA-validated.
    A = torch.empty((eff_rank, k), device=device, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
    B = torch.zeros((W_base.shape[0], eff_rank), device=device,
                    dtype=torch.float32)
    A.requires_grad_(True)
    B.requires_grad_(True)
    with torch.no_grad():
        base_product_g = X_good @ W_base.T
        base_product_b = X_bad @ W_base.T
        row_norms = W_base.norm(dim=1, keepdim=True).clamp_min(1e-8)
        W_S = W_base[:, S_idx].contiguous()
        bar_norm_sq = (
            W_base.pow(2).sum(dim=1) - W_S.pow(2).sum(dim=1)
        ).clamp_min(0.0)
        X_good_S = X_good[:, S_idx].contiguous()
        X_bad_S = X_bad[:, S_idx].contiguous()
        Y_good_dev = Y_good.to(device)
        Y_bad_dev = Y_bad.to(device)
    n_out = max(neighbor_count, 1)

    opt = LBFGS([A, B], lr=1.0, max_iter=20, history_size=10,
                line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        # Factored low-rank evaluation on the S slice only:
        #   X @ upd_full.T == (X[:, S_idx] @ A.T) @ B.T
        #   ||W_base + upd_full||_row^2 == ||W_base[:, bar_S]||_row^2 + ||W_S + B @ A||_row^2
        # Avoids materializing a full [d_out, d_in] tensor or running a dense
        # [N, d_in] @ [d_in, d_out] matmul on every L-BFGS inner evaluation.
        dW_S = B @ A
        eff_norms = (
            bar_norm_sq + (W_S + dW_S).pow(2).sum(dim=1)
        ).clamp_min(1e-16).sqrt()
        # Row-norm rescale applied to outputs as a per-column factor.
        scale = (row_norms.squeeze(1) / eff_norms).unsqueeze(0)
        new_good = (base_product_g + (X_good_S @ A.T) @ B.T) * scale
        new_bad = (base_product_b + (X_bad_S @ A.T) @ B.T) * scale
        loss_p = (new_good - Y_good_dev).pow(2).mean()
        loss_s = (_knn_mean_distances(new_bad, Y_good_dev, n_out).mean()
                  - overcorrect_weight
                  * _knn_mean_distances(new_bad, Y_bad_dev, n_out).mean())
        loss = preserve_weight * loss_p + steer_weight * loss_s
        loss.backward()
        return loss

    # Outer steps (NOT a single LBFGS call): a single strong-Wolfe call can
    # stall at the init point; outer steps converge. ARA-validated.
    for _ in range(max(n_steps, 1)):
        opt.step(closure)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        W_new = W_base.clone()
        W_new[:, S_idx] = W_S + (B @ A)
        # Restore exact base row norms (keeps unselected behavior bit-close).
        W_new = W_new * (row_norms
                         / W_new.norm(dim=1, keepdim=True).clamp_min(1e-8))
    return W_new


def select_columns(
    W_base: Tensor,
    X_good: Tensor,
    X_bad: Tensor,
    r: Tensor,
    preserve_over_steer: float,
    top_atom_fraction: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split columns into redirect set S (top fraction) and removal set.

    Returns (S_idx, bar_S_idx, rho_all) where rho is the per-column
    activation-gated shrinkage mu_b / (mu_b + (w_p/w_s) mu_g).
    """
    with torch.no_grad():
        alpha_all = W_base.T @ r
        mg = X_good.pow(2).mean(dim=0)
        mb = X_bad.pow(2).mean(dim=0)
        rho_all = mb / (mb + preserve_over_steer * mg + 1e-8)
        act_rms = (mb + 1e-8).sqrt()
        score = act_rms * alpha_all.abs()
        # heretic2 §3.3 exact: max(1, round(frac * d_in)), no upper cap.
        k = max(1, int(round(top_atom_fraction * W_base.shape[1])))
        k = min(k, W_base.shape[1])
        order = score.argsort(descending=True)
        return order[:k], order[k:], rho_all


@torch.no_grad()
def ridge_remove_columns(
    W_base: Tensor,
    S_bar_idx: Tensor,
    alpha_all: Tensor,
    rho_all: Tensor,
    r: Tensor,
    preserve_col_magnitudes: bool = True,
) -> Tensor:
    """Closed-form activation-gated removal on unselected columns."""
    W = W_base.clone()
    if S_bar_idx.numel() == 0:
        return W
    norms = W[:, S_bar_idx].norm(dim=0, keepdim=True).clamp_min(1e-8)
    beta = (alpha_all[S_bar_idx] * rho_all[S_bar_idx])
    W[:, S_bar_idx] = W[:, S_bar_idx] - torch.outer(r, beta)
    if preserve_col_magnitudes:
        W[:, S_bar_idx] = W[:, S_bar_idx] * (
            norms / W[:, S_bar_idx].norm(dim=0, keepdim=True).clamp_min(1e-8))
    return W


def _removed_optimize_column_sliced():
    """Removed Nov 2026: analytic -r target proven inert (0/12 flips at max
    steer, LBFGS stalled after 1 eval). See module header + experiments log.
    Replaced by optimize_sliced_matching. Git history preserves the code."""
