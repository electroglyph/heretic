# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA post-write mixing scan + S-span fix (experimental).
#
# Background: a per-channel affine between a write matrix and the residual
# (depthwise causal conv taps, norm weights, residual blends) rotates an
# edited output back into span(Q) — the "conv leak" (see Inkling unbound
# GATE2 §1: 19-88% of a plain removal survives). The generic fix removes
# S = span{D_k Q} (tap/weight-weighted copies of Q) instead of just Q.
#
# Cost discipline: results are cached per architecture fingerprint in
# liber_arch_registry.json. Known architectures cost one dict lookup.
# A cache miss costs ~1 structural forward for all layers; functional probe
# forwards run ONLY for structurally suspect layers (usually zero).

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

REGISTRY_PATH = os.path.join(os.path.dirname(__file__),
                             "liber_arch_registry.json")

LEAK_TOL = 0.05  # probe ratio above this means a real post-op leak


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def fingerprint_model(model: nn.Module, layers) -> str:
    """Stable id for the architecture's writer/post-op dataflow.

    Depends on code + config (class names, conv presence/shapes, bias),
    never on weight values — so every checkpoint of one architecture hits
    the same registry entry without rescanning.
    """
    parts = [_transformers_version()]
    for li, layer in enumerate(layers):
        kinds = sorted(type(m).__name__ for m in layer.modules())
        parts.append(f"{li}:{type(layer).__name__}:{','.join(kinds)}")
        for name, p in layer.named_parameters(recurse=True):
            if "conv" in name.lower() and p.ndim >= 3:
                parts.append(f"{li}:convparam:{name}:{tuple(p.shape)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_registry() -> dict:
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_registry(reg: dict) -> None:
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=1, sort_keys=True)
    os.replace(tmp, REGISTRY_PATH)


def _conv_taps(mod: nn.Module) -> Optional[List[Tensor]]:
    """Per-tap channel vectors for depthwise convs, else None."""
    if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
        w = mod.weight.detach().float().cpu()
        if mod.groups == w.shape[0] and w.shape[1] == 1:
            # depthwise [C, 1, K...]: taps are per-channel vectors
            k = w.shape[2]
            return [w[:, 0, i].reshape(-1) for i in range(k)]
    return None


def _norm_vector(mod: nn.Module) -> Optional[Tensor]:
    """Learned per-channel scale of a norm module (leaks like one conv tap)."""
    w = getattr(mod, "weight", None)
    cls = type(mod).__name__.lower()
    if "norm" in cls and isinstance(w, torch.nn.Parameter) and w.ndim == 1:
        return w.detach().float().cpu().reshape(-1)
    return None


def _is_conv_like(mod: nn.Module) -> bool:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return True
    n = type(mod).__name__.lower()
    return ("conv" in n and not isinstance(mod, nn.Linear))


def scan_post_ops(
    model: nn.Module,
    layers,
    writers_per_layer: Dict[int, List[nn.Module]],
    tokenizer,
    texts: List[str],
    batch_size: int = 1,
) -> Tuple[Dict[int, Dict[int, dict]], Dict[Tuple[int, int], object]]:
    """One-forward structural scan: which conv/norm modules consume each writer.

    Returns (specs, post_mods). specs: {layer_idx: {writer_pos: op_spec}}
    where op_spec = {"kind": "clean"|"depthwise_conv"|"norm"|"suspect",
    "taps": [[...], ...] (depthwise only), "has_bias": bool}.
    post_mods: {(layer_idx, writer_pos): the consumed module or None}.
    Only pointer-verified consumers get a concrete kind; order-only
    followers are "suspect" (recorded, never probed, never leaky).
    """
    model.eval()
    device = next(model.parameters()).device
    out_ptr: Dict[int, int] = {}
    out_shape: Dict[int, Tuple] = {}
    # Live refs: freed writer outputs get their storage recycled by later
    # modules, which would false-match pointer comparison. Holding them for
    # the duration of the forward makes every match genuine dataflow.
    out_keep: Dict[int, Tensor] = {}
    con_in: Dict[Tuple[int, int], List[int]] = {}
    order: Dict[Tuple[str, int, int], int] = {}
    tick = [0]

    def w_hook(pos: int):
        def _fn(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                out_keep[pos] = t.detach()
                out_ptr[pos] = t.data_ptr()
                out_shape[pos] = tuple(t.shape)
        return _fn

    def c_hook(li: int, ci: int):
        def _fn(_m, args, kwargs=None):
            try:
                if isinstance(args, tuple):
                    x = args[0] if len(args) > 0 else None
                    if x is None and isinstance(kwargs, dict):
                        for v in kwargs.values():
                            if torch.is_tensor(v):
                                x = v
                                break
                else:
                    x = args
                if torch.is_tensor(x):
                    con_in.setdefault((li, ci), []).append(x.data_ptr())
                    order[("conv", li, ci)] = tick[0]
                    tick[0] += 1
            except Exception:
                pass
        return _fn

    def w_order(pos: int):
        def _fn(_m, _a, _o):
            order[("writer", pos, -1)] = tick[0]
            tick[0] += 1
        return _fn

    handles = []
    conv_mods: Dict[int, List[nn.Module]] = {}
    pos_of: Dict[int, nn.Module] = {}
    pos = 0
    try:
        for li, writers in writers_per_layer.items():
            for w in writers:
                pos_of[pos] = w
                handles.append(w.register_forward_hook(w_hook(pos)))
                handles.append(w.register_forward_hook(w_order(pos)))
                pos += 1
            seen = []
            for m in layers[li].modules():
                if m is layers[li]:
                    continue
                if _is_conv_like(m) or _norm_vector(m) is not None:
                    seen.append(m)
            conv_mods[li] = seen
            for ci, m in enumerate(seen):
                handles.append(m.register_forward_pre_hook(c_hook(li, ci)))
        enc = tokenizer(texts[:batch_size], return_tensors="pt", padding=True,
                        truncation=True, return_token_type_ids=False).to(device)
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    result: Dict[int, Dict[int, dict]] = {}
    post_mods: Dict[Tuple[int, int], object] = {}
    for li, writers in writers_per_layer.items():
        result[li] = {}
        for wi, w in enumerate(writers):
            # find global pos
            gpos = next(p for p, mod in pos_of.items() if mod is w)
            spec: dict = {"kind": "clean", "taps": [], "has_bias": False}
            for ci, m in enumerate(conv_mods.get(li, [])):
                ptrs = con_in.get((li, ci), [])
                taps = _conv_taps(m)
                nv = _norm_vector(m)
                # Both required: pointer match AND the consumer runs after
                # the writer. Either alone false-matches (recycled storage
                # from a pre-write module has the same pointer but ran
                # earlier; order-only followers consume the residual sum,
                # not the isolated write).
                after = order.get(("conv", li, ci), -1) > order.get(
                    ("writer", gpos, -1), 10 ** 9)
                consumed = out_ptr.get(gpos, -1) in ptrs and after
                if taps is not None and consumed:
                    spec = {"kind": "depthwise_conv",
                            "taps": [t.tolist() for t in taps],
                            "has_bias": m.bias is not None}
                    post_mods[(li, wi)] = m
                    break
                if nv is not None and consumed:
                    spec = {"kind": "norm", "taps": [nv.tolist()],
                            "has_bias": False}
                    post_mods[(li, wi)] = m
                    break
                # Runs after the writer without consuming its output tensor:
                # recorded for transparency; the probe cannot attribute a
                # post module here, so this never triggers the S-span path.
                if (taps is not None or nv is not None) and after:
                    spec = {"kind": "suspect", "taps": [],
                            "has_bias": False}
            result[li][wi] = spec
    return result, post_mods


def build_leak_span(r_unit: Tensor,
                    op_specs: List[dict]) -> Tensor:
    """Orthonormal S = span{Q, D_k Q} over all diag post-ops.

    r_unit: (D,) refusal direction. Always includes Q itself (covers
    identity/residual blends). Returns (D, k) with k <= 1 + total taps.
    """
    vecs = [r_unit.detach().float().reshape(-1)]
    for spec in op_specs:
        for tap in spec.get("taps", []):
            d = torch.tensor(tap, dtype=torch.float32)
            if d.numel() == vecs[0].numel():
                vecs.append(d * vecs[0])
    M = torch.stack(vecs, dim=1)  # (D, m)
    Q, R = torch.linalg.qr(M)
    keep = R.abs().diag() > 1e-6 * R.abs().max().clamp_min(1e-12)
    if int(keep.sum()) == 0:
        keep[0] = True
    return Q[:, keep].contiguous()


@torch.no_grad()
def probe_leak(
    model: nn.Module,
    writer: nn.Module,
    post_mod: Optional[nn.Module],
    r_unit: Tensor,
    tokenizer,
    texts: List[str],
) -> float:
    """Hook-counterfactual leak ratio (no weight edits).

    Compares ||r^T z|| at post_mod's output with clean weights vs with the
    writer's r-component projected out in-hook. ~0 = clean. If post_mod is
    None the writer feeds the residual directly: returns 0.0.
    """
    if post_mod is None:
        return 0.0
    model.eval()
    device = next(model.parameters()).device
    r = r_unit.detach().float().reshape(-1)
    rec: Dict[str, Tensor] = {}

    def post_hook(_m, _a, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            rec.setdefault("z", []).append(t.detach().float().cpu())

    # clean pass
    h1 = post_mod.register_forward_hook(post_hook)
    enc = tokenizer(texts[:2], return_tensors="pt", padding=True,
                    truncation=True, return_token_type_ids=False).to(device)
    with torch.no_grad():
        model(**enc, use_cache=False)
    h1.remove()
    z = torch.cat(rec["z"], dim=0)
    if z.ndim == 3:
        # channel-last (B, L, D) -> last position; channel-first
        # (B, C=D, L) conv output -> last time step
        if z.shape[-1] == r.numel():
            z = z[:, -1, :]
        elif z.shape[1] == r.numel():
            z = z[:, :, -1]
        else:
            raise RuntimeError(f"cannot locate channel dim in {tuple(z.shape)}")
    z = z.reshape(-1, r.numel())
    base = float((z @ r).norm().item())
    if base < 1e-9:
        return 0.0

    # ablated pass: project r out of the writer in-hook (functional replace)
    rec.clear()
    rec["ablate"] = True  # type: ignore[assignment]

    def writer_replace(_m, _a, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            assert t.shape[-1] == r.numel(), (
                f"writer output dim {t.shape[-1]} != r dim {r.numel()}")
            y = t.detach().float()
            rd = r.to(y.device)
            proj = torch.outer(y.reshape(-1, rd.numel()) @ rd, rd)
            yn = y - proj.reshape(t.shape).to(y.dtype)
            return yn.reshape(t.shape).to(t.dtype)
        return out

    h2 = writer.register_forward_hook(writer_replace)
    h3 = post_mod.register_forward_hook(post_hook)
    with torch.no_grad():
        model(**enc, use_cache=False)
    h2.remove()
    h3.remove()
    z2 = torch.cat(rec["z"], dim=0)
    if z2.ndim == 3:
        if z2.shape[-1] == r.numel():
            z2 = z2[:, -1, :]
        elif z2.shape[1] == r.numel():
            z2 = z2[:, :, -1]
        else:
            raise RuntimeError(
                f"cannot locate channel dim in {tuple(z2.shape)}")
    leak = float((z2.reshape(-1, r.numel()) @ r).norm().item())
    return leak / base


@torch.no_grad()
def post_output_projection(
    model: nn.Module,
    post_mod: nn.Module,
    r_unit: Tensor,
    tokenizer,
    texts: List[str],
) -> float:
    """||r^T z|| at a post-write module's output, current weights, no edits.

    The verification primitive for S-span fixes: after removing S from the
    writer, this must read ~0. (Unlike probe_leak, which measures how much a
    *plain r-removal* would leak and is meaningless once S is removed.)
    """
    model.eval()
    device = next(model.parameters()).device
    r = r_unit.detach().float().reshape(-1)
    rec: Dict[str, list] = {}

    def post_hook(_m, _a, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            rec.setdefault("z", []).append(t.detach().float().cpu())

    h = post_mod.register_forward_hook(post_hook)
    try:
        enc = tokenizer(texts[:4], return_tensors="pt", padding=True,
                        truncation=True, return_token_type_ids=False).to(device)
        model(**enc, use_cache=False)
    finally:
        h.remove()
    z = torch.cat(rec["z"], dim=0)
    if z.ndim == 3:
        if z.shape[-1] == r.numel():
            z = z[:, -1, :]
        elif z.shape[1] == r.numel():
            z = z[:, :, -1]
        else:
            raise RuntimeError(
                f"cannot locate channel dim in {tuple(z.shape)}")
    return float((z.reshape(-1, r.numel()) @ r).norm().item())


@torch.no_grad()
def apply_s_ridge(
    W_base: Tensor,
    S: Tensor,
    Xg: Tensor,
    Xb: Tensor,
    a: Tensor,
    gamma: float,
    top_fraction: float,
    lambda_preserve: float,
    preserve_col_norms: bool = True,
) -> Tuple[Tensor, dict]:
    """Closed-form S-span ridge redirect for one 2D writer (CPU/GPU tensor op).

    Per column f: s_f = S^T d_f; rho_f = mb/(mb + lambda*mg);
    d_f -= rho_f * S s_f; top-(1-Q) by rms_b*||s_f|| get
    += gamma*rho_f*(a - S(S^T a)). Column norms restored (dim=0).
    """
    W = W_base.clone().float()
    StS_a = S.T @ a.reshape(-1)
    a_orth = (a.reshape(-1) - S @ StS_a)
    a_orth = a_orth / a_orth.norm().clamp_min(1e-12)
    proj = S.T @ W  # (k, d_in)
    snorm = proj.norm(dim=0)
    mg = Xg.pow(2).mean(dim=0)
    mb = Xb.pow(2).mean(dim=0)
    rho = mb / (mb + lambda_preserve * mg + 1e-8)
    score = mb.sqrt() * snorm
    k = max(1, min(W.shape[1] - 1, int(round(top_fraction * W.shape[1]))))
    order = score.argsort(descending=True)
    mask = torch.zeros(W.shape[1], dtype=torch.bool, device=W.device)
    mask[order[:k]] = True
    norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    beta = rho * snorm
    # d_f -= rho_f * S s_f  <=>  W -= S @ diag(rho) @ S^T @ W
    W = W - S @ (rho.unsqueeze(0) * proj)
    W = W + gamma * torch.outer(a_orth, beta * mask.float())
    if preserve_col_norms:
        W = W * (norms / W.norm(dim=0, keepdim=True).clamp_min(1e-8))
    return W, {"redirect_atoms": int(mask.sum().item()),
               "span_rank": int(S.shape[1])}
