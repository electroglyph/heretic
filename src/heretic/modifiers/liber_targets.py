# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA shared targets (experimental).
#
# Pure-torch helpers with no Heretic imports so they stay unit-testable:
# prefilled chat formatting, refusal direction, per-prompt acceptance
# gradients, harmlessness KL basis. Reference: heretic2 §3.1, §3.4.

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# First-token openers, read off a base model's own top next tokens. These
# sets define the acceptance score only; they never take part in atom
# selection. (Mirrors libershift/redirect.py.)
ACCEPT_WORDS = ["Sure", "Here", "Of", "Yes", "Absolutely", "Certainly",
                "Great", "Okay", "Alright"]
REFUSE_WORDS = ["I", "Sorry", "Unfortunately", "As", "No", "Cannot"]


def token_ids(tokenizer, words: List[str]) -> List[int]:
    """First token id of every word, deduplicated and sorted."""
    return sorted({tokenizer(w, add_special_tokens=False)["input_ids"][0]
                   for w in words})


def unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def project_out_subspace(V: Tensor, basis: Tensor) -> Tensor:
    """Project orthonormal basis (D, K) out of rows of V (..., D)."""
    return V - (V @ basis) @ basis.T


def format_prefill_texts(tokenizer, prompts: List[str],
                         system_prompt: str) -> dict:
    """Chat-format prompts with a prefilled empty <think></think> block.

    Both LiberShift plane fitting and ARA-style module I/O capture must
    score the same post-</think> answer-start position on reasoning models;
    an unprefilled template scores the near-constant "<think>" token and
    makes every KL look ~0. Left-pads for batched generate/forward parity.
    Callers must decode with skip_special_tokens=False to keep the prefill.
    """
    texts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if not rendered.rstrip().endswith("</think>"):
            rendered = rendered + "<think></think>\n"
        texts.append(rendered)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, return_token_type_ids=False)


def refusal_direction(mean_bad: Tensor, mean_good: Tensor) -> Tensor:
    """Unit refusal direction: bad-minus-good, orthogonalized vs good."""
    diff = mean_bad - mean_good
    g = F.normalize(mean_good, p=2, dim=1)
    diff = diff - (diff * g).sum(dim=1, keepdim=True) * g
    return F.normalize(diff, p=2, dim=1)


def per_prompt_acceptance_gradients(
    model: torch.nn.Module,
    layers,
    tokenizer,
    prompts: List[str],
    accept_ids: List[int],
    refuse_ids: List[int],
    r_unit: Tensor,
    system_prompt: str,
    harmless_bases: Optional[Dict[int, Tensor]] = None,
    grad_batch_size: int = 4,
    orth_r: bool = True,
) -> Dict[int, Tensor]:
    """Per-prompt orthogonalized, unit acceptance gradients per layer.

    `r_unit` is either a global (D,) direction (heretic2 §3.1 contract) or
    per-plane (P, D) refusal directions (LiberShift-exact: each layer's
    grads orthogonalized vs its own plane r[l+1], matching
    redirect.py:target(grad_orth) which uses r[layer+1] per block).

    Returns {layer_idx: (N_bad, D) cpu tensor}. Each row is
    d(s_i)/d(h_i[plane]) for prompt i's own acceptance score
    s = logsumexp(accept) - logsumexp(refuse), orthogonalized against the
    refusal direction (and the harmlessness basis when given), unit-normed.
    Per-prompt rows keep rank > 1 so a rank-R LoRA can steer distinct
    harmful clusters; a batch-averaged vector would collapse to rank 1
    (heretic2 §2 Theorem 5).
    """
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(layers)
    n_bad = len(prompts)
    d_model = r_unit.shape[-1]
    out: Dict[int, Tensor] = {
        l: torch.zeros((n_bad, d_model), dtype=torch.float32)
        for l in range(n_layers)
    }
    # Per-plane refusal directions when given (P, D): layer l uses plane l+1.
    # Single (D,) vector: shared global direction (heretic2 §3.1 contract).
    r_planes = None
    r_col = None
    if r_unit.ndim == 2:
        r_planes = r_unit.detach().to(device=device, dtype=torch.float32)
    else:
        r_col = r_unit.detach().to(device=device, dtype=torch.float32
                                   ).reshape(-1, 1)
    enc_all = format_prefill_texts(tokenizer, prompts, system_prompt)
    all_ids = enc_all["input_ids"]
    all_mask = enc_all["attention_mask"]

    for start in range(0, n_bad, grad_batch_size):
        end = min(start + grad_batch_size, n_bad)
        input_ids = all_ids[start:end].to(device)
        attn_mask = all_mask[start:end].to(device)
        captured: Dict[int, Tensor] = {}
        hooks = []

        def make_hook(l_idx: int):
            def _hook(_m, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                h.requires_grad_(True)
                h.retain_grad()
                captured[l_idx] = h
                return output
            return _hook

        for l_idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(l_idx)))
        try:
            model.zero_grad(set_to_none=True)
            # Heretic main() runs the whole session under
            # torch.set_grad_enabled(False) ("inference only"), so opt back
            # in here exactly like top_token_gradients does: without this,
            # the hook-set requires_grad has no graph behind it and
            # .backward() fails with "does not require grad and has no
            # grad_fn".
            with torch.enable_grad():
                logits = model(input_ids=input_ids, attention_mask=attn_mask,
                               use_cache=False).logits[:, -1, :].float()
                s = (torch.logsumexp(logits[:, accept_ids], dim=-1)
                     - torch.logsumexp(logits[:, refuse_ids], dim=-1))
                # Prompt i's graph is independent of prompt j's, so one backward
                # on the sum yields each prompt's own gradient exactly.
                s.sum().backward()
            for l_idx in range(n_layers):
                g = captured[l_idx].grad[:, -1, :].detach().float()
                # Joint orthogonalization vs span(r, U_H): build the combined
                # basis and project once. Sequential (r then U) reintroduces
                # r whenever U_H contains an r-component (it does: U_H is
                # built from raw top-1 grads, deliberately NOT pre-orth vs r).
                # QR makes the result exactly ⊥ both (Theorem 5: P_{r,U_H}).
                if orth_r:
                    if r_planes is not None and (l_idx + 1) < r_planes.shape[0]:
                        rc = r_planes[l_idx + 1].to(dtype=torch.float32)
                    elif r_planes is not None:
                        rc = r_planes.mean(dim=0).to(dtype=torch.float32)
                    else:
                        rc = r_col.reshape(-1).to(dtype=torch.float32)
                    Uh = None
                    if harmless_bases and l_idx in harmless_bases \
                            and harmless_bases[l_idx] is not None:
                        Uh = harmless_bases[l_idx].to(
                            device=device, dtype=torch.float32)  # (D, K)
                    if Uh is not None:
                        C = torch.cat([rc.reshape(-1, 1), Uh], dim=1)
                        Q, _ = torch.linalg.qr(C)
                        g = g - (g @ Q) @ Q.T
                    else:
                        g = g - (g @ rc.reshape(-1, 1)) * rc.reshape(1, -1)
                elif harmless_bases and l_idx in harmless_bases \
                        and harmless_bases[l_idx] is not None:
                    g = project_out_subspace(
                        g, harmless_bases[l_idx].to(device=device,
                                                   dtype=torch.float32))
                out[l_idx][start:end] = (g / g.norm(
                    dim=-1, keepdim=True).clamp_min(1e-8)).cpu()
        finally:
            for h in hooks:
                h.remove()
            captured.clear()
            model.zero_grad(set_to_none=True)
    return out


def harmlessness_basis(per_prompt: Tensor, rank: int) -> Tensor:
    """Top-`rank` principal directions of harmless first-token gradients.

    `per_prompt` is [prompts, n_planes, hidden]; returns [n_planes, rank,
    hidden] orthonormal rows spanning the residual directions the harmless
    next-token distribution is most sensitive to. Steering orthogonal to
    this basis cannot move that distribution to first order.
    Exact port of libershift/redirect.py:harmlessness_basis (Gram eigh,
    mean-centered, unit rows).
    """
    n_planes = per_prompt.shape[1]
    basis = torch.empty(n_planes, rank, per_prompt.shape[2])
    for plane in range(n_planes):
        x = per_prompt[:, plane, :].float()
        x = x - x.mean(dim=0, keepdim=True)
        gram = x @ x.T
        values, vectors = torch.linalg.eigh(gram.double())
        order = values.argsort(descending=True)[:rank]
        vectors = vectors[:, order].float()
        basis[plane] = unit((vectors.T @ x))
    return basis


def top_token_gradients(
    model: torch.nn.Module,
    layers,
    tokenizer,
    prompts: List[str],
    system_prompt: str,
    grad_batch_size: int = 4,
) -> Tensor:
    """Per-prompt gradient of the model's own top-1 log-prob, per layer.

    Exact port of libershift/redirect.py:top_token_gradient output contract:
    returns [prompts, n_layers, hidden] cpu tensor (one row per layer output,
    i.e. planes 1..n), last prompt position only, left-padded prefill.
    The harmlessness basis is built from THIS (not from acceptance
    gradients): the top-1 token is where the harmless distribution puts
    most mass, so these gradients span its KL-sensitive subspace
    (heretic2 §2 Theorem 4).
    """
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(layers)
    n_prompts = len(prompts)
    # hidden size from first layer probe
    rows: List[Tensor] = []
    enc_all = format_prefill_texts(tokenizer, prompts, system_prompt)
    all_ids = enc_all["input_ids"]
    all_mask = enc_all["attention_mask"]
    for start in range(0, n_prompts, grad_batch_size):
        end = min(start + grad_batch_size, n_prompts)
        input_ids = all_ids[start:end].to(device)
        attn_mask = all_mask[start:end].to(device)
        captured: Dict[int, Tensor] = {}
        hooks = []
        for l_idx, layer in enumerate(layers):
            def make_hook(li: int):
                def _fn(_m, _a, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h.requires_grad_(True)
                    h.retain_grad()
                    captured[li] = h
                    return output
                return _fn
            hooks.append(layer.register_forward_hook(make_hook(l_idx)))
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=False)
                logits = out.logits[:, -1, :].float()
                top = logits.argmax(dim=-1)
                chosen = logits.gather(1, top[:, None]).squeeze(1)
                logp = chosen - torch.logsumexp(logits, dim=-1)
                grads = torch.autograd.grad(logp.sum(), [captured[i] for i in range(n_layers)])
            batch_rows = torch.stack(
                [g[:, -1, :].detach().float().cpu() for g in grads], dim=1)
            rows.append(batch_rows)
        finally:
            for h in hooks:
                h.remove()
            captured.clear()
            model.zero_grad(set_to_none=True)
    return torch.cat(rows, dim=0)
