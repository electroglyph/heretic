# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
#
# LiberARA — LiberShift-informed ARA (experimental).
#
# Keeps stock ARA untouched. Differences from ARA, all behind this plugin:
# 1. 2D writers (o_proj, dense down_proj) get a column-sliced rank-R update:
#    only the top-(1-Q) refusal-active columns are optimized, toward the
#    good-output distribution (kNN toward good, overcorrected away from bad)
#    with isotropic preserve on good inputs and exact row-norm restoration.
#    Unselected columns stay bit-identical. See liber_dense.
#    (Nov 2026: validated on Spark-X2.6-4B; replaced the inert analytic -r
#    LBFGS core, the inert bar-column ridge removal, and the zero-effect
#    harmlessness-basis penalty. MoE support removed as unvalidated.)
# 2. All targets are computed at the prefilled <think></think> answer-start
#    position. Writers with a proven post-write leak (see liber_conv) take a
#    closed-form S-span ridge instead of the sliced match.
#
# Weight-edit mechanics: base weights are edited directly (PEFT cannot
# target fused 3D params). init() still calls model.apply_lora() so the
# export path (get_merged_model, which requires a PeftModel) keeps working:
# zero adapters merge to identity and the merged model carries the base
# edits. reset_model() restores the CPU snapshot of every edited matrix and
# zeroes adapters via model.reset_model().
# Reference: heretic2 §1–§4.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from optuna import Trial
from pydantic import BaseModel, Field, PositiveInt
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.modifier import Context, Modifier, Serializable
from heretic.plugin import load_plugin  # noqa: F401 (re-export for docs)
from heretic.utils import format_dataset_specification, print

from .liber_conv import (
    LEAK_TOL,
    apply_s_ridge,
    build_leak_span,
    fingerprint_model,
    load_registry,
    probe_leak,
    save_registry,
    scan_post_ops,
)
from .liber_dense import optimize_sliced_matching, select_columns
from .liber_targets import (
    ACCEPT_WORDS,
    REFUSE_WORDS,
    harmlessness_basis,
    per_prompt_acceptance_gradients,
    refusal_direction,
    token_ids,
    top_token_gradients,
)


@dataclass
class LiberParameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    quantile_q: float
    gamma: float
    lambda_preserve: float
    top_atom_fraction: float
    steer_weight: float
    overcorrect_weight: float = 0.9687
    neighbor_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        # Tolerant: studies started before overcorrect_weight/neighbor_count
        # existed lack those keys; fall back to defaults (fresh studies always
        # write them via suggest_parameters).
        full = {"overcorrect_weight": 0.9687, "neighbor_count": 3}
        full.update(data)
        return LiberParameters(**full)


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Prompts that tend to produce desirable responses.",
    )
    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Prompts that tend to produce undesirable responses.",
    )
    preserve_col_magnitudes: bool = Field(
        default=True,
        description="Preserve each edited write-atom column magnitude. "
        "Unlike ARA row-norm (dim=1), dim=0 column-norm never leaks into "
        "unselected columns (heretic2 §2 Theorem 3).",
    )
    lora_rank: PositiveInt = Field(
        default=16,
        description="Rank of the column-sliced B @ A_S update on 2D writers.",
    )
    grad_batch_size: PositiveInt = Field(
        default=4,
        description="Prompts per full-model backward pass.",
    )
    n_optimization_steps: PositiveInt = Field(default=5)
    skip_scan: bool = Field(
        default=False,
        description="Skip the post-write mixing scan (assume clean writers). "
        "Only use for architectures already in the registry.",
    )
    rescan: bool = Field(
        default=False,
        description="Force a fresh post-write scan even on registry hit.",
    )


class LiberARA(Modifier[LiberParameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "LiberShift-informed ARA (LiberARA, experimental)"

    # -- init: prompts, LoRA shell, residual-space targets -------------------
    def init(self, ctx: Context) -> None:
        model = ctx.get_model()
        print()
        print(f"Loading good prompts [bold]{format_dataset_specification(self.settings.good_prompts)}[/]...")
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")
        print()
        print(f"Loading bad prompts [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]...")
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")
        self.good_prompts = good_prompts
        self.bad_prompts = bad_prompts

        # Zero-adapter shell so get_merged_model() keeps working at export.
        model.apply_lora(self.settings.lora_rank)

        layers = model.get_layers()
        tok = model.tokenizer
        raw = model.model
        if isinstance(raw, torch.nn.Module) and hasattr(raw, "base_model"):
            pass  # PEFT-wrapped: hooks on get_layers() outputs still fire.
        system = good_prompts[0].system or bad_prompts[0].system or ""
        good_texts = [p.user for p in good_prompts]
        bad_texts = [p.user for p in bad_prompts]
        accept_ids = token_ids(tok, ACCEPT_WORDS)
        refuse_ids = token_ids(tok, REFUSE_WORDS)

        print("Capturing residual planes and writer I/O (once, reused across all trials)...")
        writers_by_layer = {
            li: self._writer_list(model, layers, li)
            for li in range(len(layers))
        }
        mean_good, io_good = self._mean_planes(
            raw, layers, tok, good_texts, system, writers_by_layer)
        mean_bad, io_bad = self._mean_planes(
            raw, layers, tok, bad_texts, system, writers_by_layer)
        self._writer_io: Dict[Tuple[int, int], Tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        for key, (xg, yg) in io_good.items():
            if key in io_bad:
                xb, yb = io_bad[key]
                self._writer_io[key] = (xg, xb, yg, yb)
        self.r_planes = refusal_direction(mean_bad, mean_good)  # (P, D) cpu
        self.accept_ids = accept_ids
        self.refuse_ids = refuse_ids
        self.system = system
        self._snap: Dict[str, Tensor] = {}
        self._scan_specs: Dict[int, Dict[int, dict]] = {}
        self._scan_fp = ""
        self._run_post_scan(model, layers, tok)

        # Both gradient computations below serve ONLY the leaky-writer
        # path -- clean dense models skip them entirely (top-1 grads over
        # N_good prompts was the dominant init cost). The dense
        # output-matching path needs neither.
        _need_grads = any(
            spec.get("leaky", False)
            for ws in self._scan_specs.values()
            for spec in ws.values())
        self.G_bad: Dict[int, Tensor] = {}
        self.a_mean: Dict[int, Tensor] = {}
        self.U_H = None
        if _need_grads:
            # Harmlessness basis from top-1 logprob gradients, so bad
            # gradients can be orthogonalized against it (P_{r,U_H}).
            print("Computing harmlessness basis (good, top-1)...")
            top_grads = top_token_gradients(
                raw, layers, tok, good_texts, system,
                self.settings.grad_batch_size)  # (N_good, n_layers, D)
            # Rank must be < n_prompts: centered rows sum to zero, so the
            # last Gram eigenvector is degenerate; capping avoids a
            # noise-amplified basis row.
            _rank = min(64, max(1, top_grads.shape[0] - 1))
            self.U_H = harmlessness_basis(top_grads, rank=_rank)
            # U_H[l] corresponds to layer l output = plane l+1.
            U_dict = {l: self.U_H[l].T.contiguous()  # (D, K)
                      for l in range(len(layers))}
            print("Computing per-prompt acceptance gradients (bad)...")
            self.G_bad = per_prompt_acceptance_gradients(
                raw, layers, tok, bad_texts, accept_ids, refuse_ids,
                self.r_planes, system, U_dict,
                self.settings.grad_batch_size)
            # Mean redirect target a_l = unit(mean orth grad). Each G_bad
            # row is already orth vs (r, U_H) and unit; the mean stays orth
            # but must be re-normalized to unit.
            for l in self.G_bad:
                m = self.G_bad[l].mean(dim=0)
                n = float(m.norm().item())
                self.a_mean[l] = (m / max(n, 1e-12)).cpu()
        else:
            print("LiberARA gradient targets skipped (clean dense writers: "
                  "no leaky layers).")

    def _writer_list(self, model, layers, l_idx) -> list:
        """Unwrapped 2D writers in deterministic get_layer_modules order."""
        out = []
        for _comp, modules in model.get_layer_modules(l_idx).items():
            for module in modules:
                base = getattr(module, "base_layer", module)
                if isinstance(base, nn.Linear):
                    out.append(base)
        return out

    def _run_post_scan(self, model, layers, tok) -> None:
        """Fingerprint -> registry hit (free) or one-forward scan (miss).

        Only structurally suspect layers pay for functional probe forwards.
        Clean architectures (e.g. conv-before-write) cost one forward total
        on first sight, zero afterwards.
        """
        import time as _time
        raw = model.model
        writers = {li: self._writer_list(model, layers, li)
                   for li in range(len(layers))}
        fp = fingerprint_model(raw, layers)
        self._scan_fp = fp
        reg = load_registry()
        if (not self.settings.rescan and not self.settings.skip_scan
                and fp in reg):
            print(f"LiberARA post-write scan: registry hit ({fp}) — "
                  f"verdict {reg[fp].get('verdict')}, no scan forwards.")
            self._scan_specs = {int(li): {int(wi): dict(spec)
                                          for wi, spec in ws.items()}
                                for li, ws in reg[fp].get("layers", {}).items()}
            return
        if self.settings.skip_scan:
            print("LiberARA post-write scan skipped by settings "
                  "(assuming clean writers).")
            self._scan_specs = {}
            return
        texts = self._prefill_strings(
            tok, [p.user for p in self.good_prompts[:2]], self.system)
        specs, post_mods = scan_post_ops(
            raw, layers, writers, tok, texts, batch_size=1)
        # Functional probe ONLY for pointer-verified consumers
        # (depthwise_conv/norm kinds). "suspect" entries have no attributed
        # post module, so probing them would measure an unrelated module.
        probes: Dict[str, float] = {}
        for li, ws in specs.items():
            for wi, spec in ws.items():
                if spec["kind"] not in ("depthwise_conv", "norm"):
                    continue
                wmod = writers[li][wi]
                post = post_mods.get((li, wi))
                r_l = self.r_planes[li + 1]
                ratio = probe_leak(raw, wmod, post, r_l, tok, texts)
                probes[f"{li}:{wi}"] = ratio
                if ratio > LEAK_TOL:
                    ws[wi] = dict(spec, probed_ratio=ratio, leaky=True)
        self._scan_specs = specs
        any_leaky = any(spec.get("leaky", False)
                        for ws in specs.values() for spec in ws.values())
        reg[fp] = {"verdict": "leaky" if any_leaky else "clean",
                   "layers": {str(li): {str(wi): spec
                                        for wi, spec in ws.items()}
                              for li, ws in specs.items()},
                   "probes": probes,
                   "date": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())}
        try:
            save_registry(reg)
            print(f"LiberARA post-write scan: miss ({fp}) — 1 structural "
                  f"forward, {len(probes)} probe(s), verdict "
                  f"{reg[fp]['verdict']}. Recorded.")
        except OSError as exc:
            print(f"LiberARA post-write scan: registry write failed ({exc}); "
                  f"continuing without cache.")

    def _mean_planes(self, raw, layers, tok, texts, system,
                     writers_by_layer=None):
        from .liber_targets import format_prefill_texts
        device = next(raw.parameters()).device
        P, D = len(layers) + 1, self._hidden(raw, layers)
        total = torch.zeros(P, D, dtype=torch.float64)
        n = 0
        buf: Dict[int, Tensor] = {}
        last: Dict[str, Tensor] = {}
        w_xs: Dict[Tuple[int, int], list] = {}
        w_ys: Dict[Tuple[int, int], list] = {}
        handles = []

        def pre(i: int):
            def _fn(_m, args):
                x = args[0] if isinstance(args, tuple) else args
                buf[i] = x[:, -1, :].detach().float().cpu()
            return _fn

        def post(_m, _a, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                last["h"] = t[:, -1, :].detach().float().cpu()

        def w_hook(key: Tuple[int, int]):
            def _fn(_m, args, out):
                x = args[0] if isinstance(args, tuple) else args
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(x) and torch.is_tensor(t):
                    # Store in bfloat16 on CPU (~1.8 GB for 1400 prompts x 72
                    # writers); cast to float32 on device per module in modify_model.
                    w_xs.setdefault(key, []).append(
                        x[:, -1, :].detach().to(torch.bfloat16).cpu())
                    w_ys.setdefault(key, []).append(
                        t[:, -1, :].detach().to(torch.bfloat16).cpu())
            return _fn

        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(pre(i)))
        handles.append(layers[-1].register_forward_hook(post))
        if writers_by_layer:
            for li, wlist in writers_by_layer.items():
                for wi, wmod in enumerate(wlist):
                    if wmod.weight.shape[0] == D:
                        handles.append(
                            wmod.register_forward_hook(w_hook((li, wi))))
        try:
            for s in range(0, len(texts), 32):
                buf.clear()
                last.clear()
                enc = format_prefill_texts(tok, texts[s:s + 32], system)
                with torch.no_grad():
                    raw(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        use_cache=False)
                bsz = buf[0].shape[0]
                planes = torch.empty(bsz, P, D)
                for ell in range(len(layers)):
                    planes[:, ell, :] = buf[ell][:bsz]
                planes[:, -1, :] = last["h"][:bsz]
                total += planes.double().sum(0)
                n += bsz
        finally:
            for h in handles:
                h.remove()
        mean_p = (total / max(n, 1)).float()
        if writers_by_layer is None:
            return mean_p
        io_out = {
            k: (torch.cat(w_xs[k], dim=0), torch.cat(w_ys[k], dim=0))
            for k in w_xs
        }
        return mean_p, io_out

    @staticmethod
    def _hidden(raw, layers) -> int:
        cfg = getattr(raw, "config", None)
        # PEFT wraps the base model; hidden_size lives on the base config.
        base = getattr(raw, "base_model", None)
        if cfg is None and base is not None:
            base_model = getattr(base, "model", None)
            cfg = getattr(base_model, "config", None)
        if cfg is not None and getattr(cfg, "hidden_size", None):
            return int(cfg.hidden_size)
        for layer in layers:
            for _, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    return mod.out_features
        raise RuntimeError("cannot infer hidden size")

    # -- trial parameters ----------------------------------------------------
    def suggest_parameters(self, ctx: Context, trial: Trial) -> LiberParameters:
        layer_count = len(ctx.get_model().get_layers())
        start = trial.suggest_int("start_layer_index", 0, layer_count // 2)
        end = trial.suggest_int("end_layer_index", layer_count // 2 + 1,
                                layer_count)
        return LiberParameters(
            start_layer_index=start,
            end_layer_index=end,
            quantile_q=trial.suggest_float("quantile_q", 0.90, 0.94),
            gamma=trial.suggest_float("gamma", 0.45, 0.65),
            lambda_preserve=trial.suggest_float("lambda_preserve", 0.5, 1.0),
            top_atom_fraction=trial.suggest_float(
                "top_atom_fraction", 0.05, 0.15),
            steer_weight=trial.suggest_float("steer_weight", 0.0001, 1.0,
                                             log=True),
            overcorrect_weight=trial.suggest_float(
                "overcorrect_weight", 0.0, 1.3),
            neighbor_count=trial.suggest_int(
                "neighbor_count", 1,
                min(15, len(self.good_prompts), len(self.bad_prompts))),
        )

    # -- edit ----------------------------------------------------------------
    def modify_model(self, ctx: Context, parameters: LiberParameters) -> None:
        model = ctx.get_model()
        layers = model.get_layers()
        raw = model.model
        tok = model.tokenizer
        device = next(raw.parameters()).device
        layer_range = range(parameters.start_layer_index,
                            parameters.end_layer_index)
        self._snap = {}
        r = self.r_planes.to(device=device, dtype=torch.float32)

        # Column-sliced output matching on 2D writers, or S-span closed-form
        # ridge on writers with a proven post-write leak.
        for l_idx in layer_range:
            r_l = r[l_idx + 1]
            for wi, base_mod in enumerate(self._writer_list(
                    model, layers, l_idx)):
                if base_mod.weight.shape[0] != r_l.shape[0]:
                    continue  # not a residual writer (e.g. gate/up)
                spec = self._scan_specs.get(l_idx, {}).get(wi, {})
                base = base_mod.weight.detach().float()
                cached_io = getattr(self, "_writer_io", {}).get((l_idx, wi))
                if cached_io is not None:
                    Xg = cached_io[0].float().to(device)
                    Xb = cached_io[1].float().to(device)
                    Yg = cached_io[2].float().to(device)
                    Yb = cached_io[3].float().to(device)
                else:
                    Xg, Xb, Yg, Yb = self._last_position_inputs(
                        raw, base_mod, tok, l_idx, device)
                if Xg is None:
                    continue
                self._snapshot(base_mod)
                if spec.get("leaky"):
                    a_l = self.a_mean[l_idx].to(device=device,
                                                dtype=torch.float32)
                    S = build_leak_span(r_l.cpu(), [spec]).to(device)
                    a_orth = a_l - S @ (S.T @ a_l)
                    W_new, info = apply_s_ridge(
                        base.to(device), S, Xg, Xb, a_orth,
                        parameters.gamma, parameters.top_atom_fraction,
                        parameters.lambda_preserve,
                        self.settings.preserve_col_magnitudes)
                    print(f"LiberARA S-ridge l{l_idx}w{wi}: {info}")
                else:
                    # Validated dense path (Nov 2026): sliced output matching.
                    # The old analytic -r LBFGS core was proven inert (0/12
                    # flips at max steer; LBFGS stalled after 1 eval), the
                    # U_H basis penalty had zero effect at any rank/weight,
                    # and bar-column ridge removal added nothing measurable.
                    S_idx, _, _ = select_columns(
                        base.to(device), Xg, Xb, r_l.to(device),
                        1.0 / max(parameters.steer_weight, 1e-8),
                        parameters.top_atom_fraction)
                    W_new = optimize_sliced_matching(
                        base.to(device), Xg, Xb, Yg, Yb, S_idx,
                        parameters.lambda_preserve,
                        parameters.steer_weight,
                        parameters.overcorrect_weight,
                        parameters.neighbor_count,
                        self.settings.lora_rank,
                        self.settings.n_optimization_steps)
                with torch.no_grad():
                    base_mod.weight.data.copy_(
                        W_new.to(base_mod.weight.dtype))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefill_strings(self, tok, prompts, system) -> list:
        # Same contract as liber_targets.format_prefill_texts: left-padded
        # prefill so [:, -1, :] is the answer-start position on every prompt.
        prev_side = getattr(tok, "padding_side", None)
        tok.padding_side = "left"
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token_id = tok.eos_token_id
        try:
            out = []
            for p in prompts:
                t = tok.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
                if not t.rstrip().endswith("</think>"):
                    t = t + "<think></think>\n"
                out.append(t)
        finally:
            if prev_side is not None:
                tok.padding_side = prev_side
        return out

    def _last_position_inputs(self, raw, module, tok, l_idx, device):
        """Last-position inputs AND outputs of one Linear for good and bad."""
        good_t = self._prefill_strings(
            tok, [p.user for p in self.good_prompts], self.system)
        bad_t = self._prefill_strings(
            tok, [p.user for p in self.bad_prompts], self.system)
        acc: Dict[str, Tensor] = {}

        def pre_hook(_m, args):
            x = args[0] if isinstance(args, tuple) else args
            acc.setdefault("xs", []).append(
                x[:, -1, :].detach().float().cpu())

        def post_hook(_m, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                acc.setdefault("ys", []).append(
                    t[:, -1, :].detach().float().cpu())

        outs = {}
        prev_side = getattr(tok, "padding_side", None)
        tok.padding_side = "left"
        try:
            for key, texts in (("good", good_t), ("bad", bad_t)):
                acc.clear()
                h1 = module.register_forward_pre_hook(pre_hook)
                h2 = module.register_forward_hook(post_hook)
                try:
                    for s in range(0, len(texts), 8):
                        enc = tok(texts[s:s + 8], return_tensors="pt",
                                  padding=True, truncation=True,
                                  return_token_type_ids=False).to(device)
                        with torch.no_grad():
                            raw(**enc, use_cache=False)
                finally:
                    h1.remove()
                    h2.remove()
                outs[key] = (torch.cat(acc["xs"], dim=0).to(device),
                             torch.cat(acc["ys"], dim=0).to(device))
        finally:
            if prev_side is not None:
                tok.padding_side = prev_side
        return outs["good"][0], outs["bad"][0], outs["good"][1], outs["bad"][1]

    def _snapshot(self, module: nn.Linear) -> None:
        key = f"lin:{id(module)}"
        if key not in self._snap:
            self._snap[key] = module.weight.detach().cpu().clone()
            self._snap[key + ":ref"] = module.weight

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        with torch.no_grad():
            for key, saved in self._snap.items():
                if key.endswith(":ref"):
                    continue
                ref = self._snap.get(key + ":ref")
                if ref is not None:
                    ref.copy_(saved.to(ref.device, ref.dtype))
        self._snap = {}
        # Zero the (unused) adapter shell so the next trial starts clean,
        # mirroring ARA's fast path.
        model.reset_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
