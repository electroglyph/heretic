# liber branch plan — experimental LiberShift-informed ARA for Heretic

## 0. Goal and non-goals

- Goal: add LiberShift-informed steering as **experimental, opt-in** Heretic modifiers on a new branch `liber`, proving refusal/KL gains on MoE models (esp. LFM2.5-24B-A2B) without touching stock behavior.
- Non-goals: no changes to stock ARA results, default configs, existing tests, or published model cards. `master` stays green; `liber` is where everything new lives.
- Proofs, theorems, and reference implementations this plan builds on live in `/home/anon/LiberShift/heretic2.md` (cited as `heretic2 §X`, not duplicated here).

## 1. Starting state (verified)

- Repo: `/home/anon/heretic`, `master` at `aecd4a8`, clean (`git status` shows `## master...origin/master`).
- Stock ARA: `src/heretic/modifiers/ara.py` (`ara_loss`, per-module LoRA + LBFGS, row-norm `dim=1`, `optimizer.zero_grad(set_to_none=True)`).
- Modifier contract: `src/heretic/modifier.py` (`suggest_parameters` / `modify_model` / `reset_model`); loader enforces exactly one modifier per run (`load_and_init_modifiers`); selection is by config plugin string (`config.default.toml:87` → `{ plugin = "heretic.modifiers.ara.ARA" }`).
- Module discovery: `src/heretic/model.py:get_layer_modules` iterates **per-expert `nn.Linear`s**. LFM2.5-24B-A2B's 38 MoE layers store experts as **fused 3D params** (`down_proj [64, 2048, 1536]`), so none of the per-expert branches match — stock ARA edits mixers + 2 dense layers only. (Consequence of `heretic2 §1 Theorem 1`: ~97.25% of write atoms unaddressed. See `heretic2 §1` for the atom accounting; not repeated here.)
- Tests: `tests/test_ara_lowrank_equivalence.py` guards ARA's low-rank reformulation; model dirs under `tests/` pin configs.

## 2. Branch and compatibility rules

1. `git checkout -b liber` from `master`. All work below happens on `liber`; nothing lands on `master` until gates (§6) pass.
2. **Additive only**: new files + one new example config. Forbidden on `liber` until acceptance: edits to `modifiers/ara.py`, `config.default.toml`, existing scorers, `model.py` discovery, or any file under `tests/` except new `test_liber_*` files.
3. Stock path stays byte-identical: default config still points at `heretic.modifiers.ara.ARA`; CI runs the same suite. The new modifier is selected only by pointing `modifiers = [{ plugin = "heretic.modifiers.liber.LiberARA" }]` plus a `[modifiers.LiberARA]` settings table (see §3 `config.liber.toml` row — the table is load-bearing, not optional).
4. Reproducibility: new modifier sets `reproducible = True` and seeds exactly like ARA (per-layer generator from `heretic_settings.seed`). Stays upload-reproducible because `is_builtin_plugin` keys on the `heretic.` prefix, which the new `heretic.modifiers.liber.*` modules keep.

## 3. What to add (new files on `liber`)

| File | Contents | Built from |
| ---- | -------- | ---------- |
| `src/heretic/modifiers/liber.py` | `LiberARA(Modifier[LiberParameters])` + `LiberParameters(Serializable)` dataclass (mirroring ARA's `Parameters`: `to_dict` / `to_presentation_dict` / `from_dict` so `render_trial_parameters` works unmodified): `Settings` (superset of ARA knobs + `quantile_q`, `gamma`, `lambda_preserve`, `top_atom_fraction`, `lora_rank`, `w_kl_basis`, `use_moe_ridge`, `preserve_col_magnitudes`); `suggest_parameters` (layer range + new knobs with narrow ranges); `modify_model` (Stage 2 → Stage 3 orchestration); `reset_model` (mirror ARA exactly: `model.reset_model()` fast path, else re-apply LoRA) | `heretic2 §4` stages; ARA `modify_model` loop shape |
| `src/heretic/modifiers/liber_targets.py` | Prefill helper (`<think></think>` formatting), per-prompt acceptance gradients $\hat G$ (orthogonalized vs $r$, $U_H$), harmlessness basis $U_H$ capture | `heretic2 §3.1`, `heretic2 §3.4` |
| `src/heretic/modifiers/liber_moe.py` | Closed-form activation-aware ridge redirect for fused 3D `down_proj [E, D, I]` (moment capture + $\beta^*$ + column-norm `dim=1`-of-`[E,D,I]`) | `heretic2 §3.2`, `heretic2 §2.4/Theorem 6` |
| `src/heretic/modifiers/liber_dense.py` | Column-sliced multi-rank LoRA for 2D writers (mixers + dense `down_proj`): top-$(1-Q)$ slice $S$, closed-form removal on $\bar S$, LBFGS on $B A_S$ with `dim=0` column-norm + KL-basis penalty + per-prompt target | `heretic2 §3.3`, `heretic2 §2.1–2.3` |
| `config.liber.toml` | Copy of `config.default.toml` with the `modifiers` entry swapped to the new plugin **plus** a `[modifiers.LiberARA]` table carrying the starting grid (narrow layer range, `quantile_q`/`gamma`/`lambda_preserve`/`lora_rank`/`w_kl_basis`). The table is required: `Plugin.get_settings_raw` (`plugin.py:307-341`) reads `[<top>.ClassName]` (`modifiers.LiberARA`) and silently falls back to schema defaults for any missing key, so a plugin-string-only copy would run all-defaults. | `heretic2 §4` gates |
| `tests/test_liber_equivalence.py` | Column-sliced low-rank equivalence in the repo's `unittest.TestCase` style (mirrors `test_ara_lowrank_equivalence.py` for the $S$-slice formulation) | ARA test pattern |
| `tests/test_liber_moe.py` | Shape/dtype/seed determinism + column-norm preservation on a fake fused `[E, D, I]` module; sparse-routing (empty-expert) no-crash | `heretic2 §2.4` crash mode |

No other files. The `transformers` MoE dispatch patch (`heretic2 §1 Code 1`) stays **out of the repo** on this branch: it is a documented manual pre-step on the 24B machine only, not a Heretic source change and not a new util file.

## 4. Integration points (how stock stays untouched)

- **Selection, not replacement**: `plugin.load_plugin` already resolves `heretic.modifiers.liber.LiberARA`; stock runs never import the new modules. Because the new modules live under `heretic.*`, `is_builtin_plugin` (`plugin.py:55-63`, `name.startswith("heretic.")`) still classifies them as built-in, so the reproducibility offer on upload is preserved.
- **Discovery reuse + one new path**: dense/mixer layers reuse `model.get_layer_modules()` read-only. Fused 3D MoE is handled by duck-typing (`gate_up_proj` + 3D `down_proj`) inside `liber_moe.py` only — no edits to `model.py`. Custom MoE hooks must tolerate experts absent in a given batch (contribute zero, never crash), matching the sparse-routing contract `get_module_io_batched` (`model.py:873-930`) already implements.
- **Bypass managed response/logit helpers for prefill-sensitive paths**: `ctx.get_responses` hardcodes `skip_special_tokens=True` (`plugin.py:173-180`), which destroys the `<think></think>` prefill structure the whole comparison depends on. All prefill-sensitive capture and generation in the new modules must use `ctx.get_model()` directly with `skip_special_tokens=False` and the shared prefill formatter — never the managed helpers.
- **I/O capture discipline**: same `get_module_io_batched`-style capture at the prefilled answer position for both old and new paths; new grads/basis captured in the new modules only.
- **Scorers unchanged**: keyword-rate / KL / benchmark scorers are not modified; eval batching stays whatever the run config sets (note `heretic2 §4 Stage 0` lock when comparing numbers).

## 5. OOM and eval discipline (from hard-won remote lessons)

- Keep the fused-gather fix out of the hot path: single-prompt or small-batch generation + `use_cache=True` for decode loops; `empty_cache()` between phases; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before first CUDA alloc. Rationale: `heretic2 §1 Theorem 2`.
- Never share one work/checkpoint dir between concurrent runs; kernel restart after a VRAM-pinning crash before reloading.
- Gate numbers exactly as `heretic2 §4`: Stage 2 must beat stock at equal-or-lower refusals with lower KL/PPL; Stage 3 keeps only with ≥4/100 refusal drop at ΔKL ≤ +0.03 and ΔPPL ≤ +0.10.

## 6. Acceptance gates (all must pass before any merge proposal)

1. `pytest tests/` green on `liber`, including all pre-existing tests unmodified.
2. New `test_liber_*` green + determinism check (same seed → identical weights).
3. Small-model e2e (tiny/Qwen fixture): stock ARA config output unchanged vs `master`; `config.liber.toml` run completes and logs both gates.
4. 24B decision (remote, persistent work dir): Stage 2/3 table filled per `heretic2 §6` template; keep/drop call recorded. Negative results kept.

## 7. Ordered work steps

1. `git checkout -b liber && git status` (clean base).
2. Add `liber_targets.py` (prefill + $\hat G$ + $U_H$) + its unit test.
3. Add `liber_moe.py` + `test_liber_moe.py` (closed-form ridge; proves the 97% coverage gap closed).
4. Add `liber_dense.py` + `test_liber_equivalence.py` (column-sliced LoRA; proves zero leakage into $\bar S`).
5. Add `liber.py` (`LiberARA`) wiring Stage 2 → recapture → Stage 3 with the two gates.
6. Add `config.liber.toml` (default config untouched).
7. Run gates §6.1–6.3 locally; then the 24B table (§6.4) on the big card.

## 8. Where the math lives (reference, not copy)

- Atom decomposition + MoE OOM fix: `heretic2 §1` (Theorems 1–2, Code 1).
- Column slicing, KL basis, per-prompt rank, MoE ridge: `heretic2 §2` (Theorems 3–6).
- Drop-in reference code adapted by §3 files: `heretic2 §3.1–3.4`.
- Stage order and gate values: `heretic2 §4`; corrections log: `heretic2 §5`; decision table: `heretic2 §6`.
