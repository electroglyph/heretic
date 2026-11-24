# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA fused-MoE closed-form ridge redirect (experimental).
#
# Stock ARA's get_layer_modules() only sees per-expert nn.Linear modules, so
# fused 3D experts (e.g. LFM2.5 down_proj [E, D, I], ~97% of write atoms) are
# never edited — and stock mean-knn steering crashes on sparse routing when
# an expert sees fewer tokens than neighbor_count. This module replaces both
# with a gate-weighted closed-form ridge solve per atom column, in one
# streaming pass with ~48 MB of accumulators and zero LBFGS overhead.
# Reference: heretic2 §2.4 (Theorem 6), §3.2.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def find_fused_moe(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return the fused-experts submodule if present, else None.

    Matches modules holding gate_up_proj [E, 2*I, D] and down_proj [E, D, I]
    (Lfm2MoeExperts-style fused 3D parameters).
    """
    for sub in module.modules():
        gate = getattr(sub, "gate_up_proj", None)
        down = getattr(sub, "down_proj", None)
        if gate is not None and down is not None \
                and isinstance(down, torch.nn.Parameter) and down.ndim == 3:
            return sub
    return None


@torch.no_grad()
def collect_moe_moments(
    model: torch.nn.Module,
    layers,
    tokenizer,
    texts: List[str],
    layer_range: range,
    batch_size: int = 4,
) -> Tuple[Dict[int, Tensor], int]:
    """Gate-weighted sum of squared expert activations per (layer, expert, col).

    Returns ({layer_idx: (E, I) cpu}, n_prompts). Hooks the fused experts
    forward pre-hook; experts receiving zero tokens in a batch contribute
    zero (never crash on sparse routing). Only the last prompt position is
    accumulated (the prefilled answer-start position).
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    # Last-position accumulation requires left padding (answer-start at -1).
    try:
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass
    in_range = set(layer_range)
    moe_of: Dict[int, torch.nn.Module] = {}
    for l_idx in in_range:
        if 0 <= l_idx < len(layers):
            found = find_fused_moe(layers[l_idx])
            if found is not None:
                moe_of[l_idx] = found
    acc: Dict[int, Tensor] = {}
    hooks = []

    def make_hook(l_idx: int, exp_mod: torch.nn.Module):
        def _pre(_m, args):
            # args: (hidden_states, top_k_index, top_k_weights); shapes vary
            # by dispatcher, so normalize defensively and skip on mismatch.
            try:
                h, tki, tkw = args[0], args[1], args[2]
                h = h.detach().float()
                if h.ndim == 3:
                    B, L, D = h.shape
                    x_last = h[:, -1, :]
                    idx = tki.reshape(B, L, -1)[:, -1, :]
                    w = tkw.reshape(B, L, -1)[:, -1, :]
                elif h.ndim == 2:
                    x_last = h
                    idx = tki.reshape(h.shape[0], -1)
                    w = tkw.reshape(h.shape[0], -1)
                else:
                    return
                D = x_last.shape[-1]
                inter = exp_mod.down_proj.shape[2]
                x_cpu = x_last.cpu()
                idx_cpu = idx.cpu()
                w_cpu = w.float().cpu()
                dev = exp_mod.gate_up_proj.device
                for e in torch.unique(idx_cpu).tolist():
                    e = int(e)
                    if e < 0 or e >= exp_mod.down_proj.shape[0]:
                        continue
                    rows, slots = torch.where(idx_cpu == e)
                    if rows.numel() == 0:
                        continue
                    xe = x_cpu[rows].to(dev)
                    with torch.no_grad():
                        # gate_up_proj may be bf16/fp16 (dtype auto); match
                        # activation dtype or F.linear raises (and the bare
                        # except below would silently zero the moments).
                        gu = F.linear(xe, exp_mod.gate_up_proj[e].to(xe.dtype))
                    gate, up = gu.split(inter, dim=-1)
                    act = (F.silu(gate) * up).float().cpu()
                    ge = w_cpu[rows, slots].unsqueeze(-1)
                    acc[l_idx][e] += ((ge * act) ** 2).sum(dim=0)
            except (KeyError, IndexError, ValueError):
                # Unknown dispatcher shapes: skip this batch for this layer.
                # Deliberately NOT bare `except`: a dtype/device RuntimeError
                # here once silently zeroed every moment (bf16 weights vs
                # float32 activations). Real failures must surface.
                return
        return _pre

    for l_idx, exp_mod in moe_of.items():
        E, _, I = exp_mod.down_proj.shape
        acc[l_idx] = torch.zeros((E, I), dtype=torch.float64)
        hooks.append(exp_mod.register_forward_pre_hook(
            make_hook(l_idx, exp_mod)))
    n = 0
    try:
        for s in range(0, len(texts), batch_size):
            enc = tokenizer(texts[s:s + batch_size], return_tensors="pt",
                            padding=True, truncation=True,
                            return_token_type_ids=False).to(device)
            with torch.no_grad():
                model(**enc, use_cache=False)
            n += enc["input_ids"].shape[0]
    finally:
        for h in hooks:
            h.remove()
    return ({k: v.float().cpu() for k, v in acc.items()}, max(n, 1))


@torch.no_grad()
def apply_moe_ridge(
    model: torch.nn.Module,
    layers,
    mu_good: Dict[int, Tensor],
    count_good: int,
    mu_bad: Dict[int, Tensor],
    count_bad: int,
    r_unit: Tensor,
    a_layers: Dict[int, Tensor],
    layer_range: range,
    quantile_q: float = 0.92,
    gamma: float = 0.50,
    lambda_preserve: float = 1.0,
    preserve_col_norms: bool = True,
) -> Dict[str, float]:
    """Apply the closed-form activation-aware ridge redirect in place.

    For each fused expert column d (atom): beta = rho * (d . r) with
    rho = mu_b / (mu_b + lambda * mu_g); W <- W - beta r^T (+ gamma beta a^T
    on top-(1-Q) atoms by |beta|); column norms (dim=1 of [E, D, I])
    restored. Harmless-specialist experts (mu_b << mu_g) get rho -> 0 and
    are spared. Returns edit statistics.
    """
    beta_dict: Dict[int, Tensor] = {}
    scores = []
    for l_idx in layer_range:
        if l_idx not in mu_good or l_idx not in mu_bad:
            continue
        exp_mod = None
        if 0 <= l_idx < len(layers):
            exp_mod = find_fused_moe(layers[l_idx])
        if exp_mod is None:
            continue
        dev = exp_mod.down_proj.device
        W = exp_mod.down_proj.data.float().to(dev)
        dev = exp_mod.down_proj.device
        r = r_unit.detach().to(device=dev, dtype=torch.float32)
        mg = mu_good[l_idx].to(dev) / max(count_good, 1)
        mb = mu_bad[l_idx].to(dev) / max(count_bad, 1)
        rho = mb / (mb + lambda_preserve * mg + 1e-8)
        alpha = torch.einsum("edi,d->ei", W, r)
        beta = rho * alpha
        beta_dict[l_idx] = beta
        scores.append(beta.abs().reshape(-1))
    if not scores:
        return {"moe_layers_edited": 0, "redirect_atoms": 0, "threshold": 0.0}
    flat = torch.cat(scores)
    threshold = torch.quantile(flat, quantile_q).item()
    edited = 0
    redirect_atoms = 0
    for l_idx, beta in beta_dict.items():
        exp_mod = find_fused_moe(layers[l_idx])
        dev = exp_mod.down_proj.device
        r = r_unit.detach().to(device=dev, dtype=torch.float32)
        W = exp_mod.down_proj.data.float().to(dev)
        orig_norms = W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        a_l = a_layers[l_idx].detach().to(device=dev, dtype=torch.float32)
        W_new = W - torch.einsum("ei,d->edi", beta, r)
        mask = (beta.abs() >= threshold).float()
        redirect_atoms += int(mask.sum().item())
        if gamma and mask.any():
            W_new = W_new + gamma * torch.einsum(
                "ei,d->edi", beta * mask, a_l)
        if preserve_col_norms:
            W_new = W_new * (orig_norms
                             / W_new.norm(dim=1, keepdim=True).clamp_min(1e-8))
        exp_mod.down_proj.data.copy_(W_new.to(exp_mod.down_proj.dtype))
        edited += 1
    return {"moe_layers_edited": edited, "redirect_atoms": redirect_atoms,
            "threshold": threshold}
