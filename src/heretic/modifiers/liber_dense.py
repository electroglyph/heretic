# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA column-sliced multi-rank LoRA for 2D writer matrices (experimental).
#
# Two corrections to stock ARA on write atoms (columns):
# 1. Column slicing: only the top-(1-Q) refusal-active columns S are LoRA'd;
#    the rest get closed-form ridge removal. Stock ARA's row-norm (dim=1)
#    rescaling leaks into unselected columns; dim=0 column-norm keeps them
#    bit-identical while preserving each selected atom's magnitude.
# 2. Per-prompt acceptance-gradient target with the refusal component
#    subtracted (stock mean-vector target keeps 100% of the refusal since
#    a_l ⊥ r) plus a harmlessness-basis KL penalty instead of isotropic MSE
#    alone. Reference: heretic2 §2.1–2.3, §3.3.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim import LBFGS


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


def optimize_column_sliced(
    W_base: Tensor,
    X_good: Tensor,
    X_bad: Tensor,
    r: Tensor,
    G_bad: Tensor,
    U_H: Optional[Tensor],
    S_idx: Tensor,
    alpha_S: Tensor,
    gamma: float,
    lora_rank: int,
    w_preserve: float,
    w_kl_basis: float,
    w_steer: float,
    n_steps: int,
    seed: int = 42,
    preserve_col_magnitudes: bool = True,
) -> Tensor:
    """Rank-R LBFGS on the column slice S toward the per-prompt target.

    Target shift on bad inputs: -alpha_tok r^T + gamma alpha_tok G_bad.
    Loss: w_preserve * (isotropic MSE + w_kl_basis * basis-projected MSE)
    + w_steer * MSE(delta_Y_bad, target). Column norms (dim=0) restored.
    Returns the full updated weight (only S columns changed).
    """
    device = W_base.device
    W_S_base = W_base[:, S_idx].clone()
    col_norms = W_S_base.norm(dim=0, keepdim=True).clamp_min(1e-8)
    Xg = X_good[:, S_idx]
    Xb = X_bad[:, S_idx]
    k = S_idx.numel()
    eff_rank = max(1, min(lora_rank, k, W_base.shape[0]))
    gen = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn((eff_rank, k), generator=gen, device=device,
                    dtype=torch.float32) * 0.01
    B = torch.zeros((W_base.shape[0], eff_rank), device=device,
                    dtype=torch.float32)
    A.requires_grad_(True)
    B.requires_grad_(True)
    with torch.no_grad():
        alpha_tok = (Xb @ alpha_S).unsqueeze(-1)
        target = (-alpha_tok * r.unsqueeze(0)
                  + gamma * alpha_tok * G_bad.to(device))
    # heretic2 §3.3 exact: single LBFGS step with max_iter=n_steps
    # (strong_wolfe, history 10, lr 1.0). NOT an outer loop of steps.
    opt = LBFGS([A, B], lr=1.0, max_iter=max(n_steps, 1), history_size=10,
                line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        W_S = W_S_base + (B @ A)
        if preserve_col_magnitudes:
            W_S = W_S * (col_norms
                         / W_S.norm(dim=0, keepdim=True).clamp_min(1e-8))
        dW = W_S - W_S_base
        dYg = Xg @ dW.T
        dYb = Xb @ dW.T
        loss_p = dYg.pow(2).mean()
        if U_H is not None and w_kl_basis > 0:
            loss_p = loss_p + w_kl_basis * ((dYg @ U_H.to(device)).pow(2)
                                           .mean())
        loss_s = torch.nn.functional.mse_loss(dYb, target)
        loss = w_preserve * loss_p + w_steer * loss_s
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        W_S = W_S_base + (B @ A)
        if preserve_col_magnitudes:
            W_S = W_S * (col_norms
                         / W_S.norm(dim=0, keepdim=True).clamp_min(1e-8))
        W_new = W_base.clone()
        W_new[:, S_idx] = W_S
    return W_new
