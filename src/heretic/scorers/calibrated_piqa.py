# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Calibrated PIQA scorer: rank candidates on base-model flip-flopper questions.

Instead of scoring every PIQA question once (like BenchmarkScore), this scorer
finds the questions whose outcome is uncertain under the base model (split
decision across repeated generative A/B samples), then scores every candidate
only on that frozen flip-flopper set.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import SingleDatasetSpecification
from heretic.model import strftime_fixed_date
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import Prompt, print

# Offset separating the discovery seed stream (runs 0..2) from the scoring
# seed stream (runs 1000..1004). Sharing seeds would redraw discovery's exact
# outcomes during scoring (same seed + same weights + same prompt = same draw).
SCORING_RUN_OFFSET = 1000

# Filename (inside study_checkpoint_dir) for the flip set + baseline record.
CACHE_FILENAME = "calibrated_piqa.json"


class Settings(BaseModel):
    score_name: str = Field(
        default="Calibrated PIQA",
        description="Name that describes what this scorer measures.",
    )

    dataset: SingleDatasetSpecification = Field(
        # default_factory: the commit hash is mutated in at load time, so each
        # scorer instance needs its own specification object.
        default_factory=lambda: SingleDatasetSpecification(
            dataset="baber/piqa",
            split="validation",
        ),
        description="PIQA dataset. Rows are read manually (goal/sol1/sol2/label); "
        "no single text column exists, so ctx.load_prompts cannot be used.",
    )

    discovery_runs: int = Field(
        default=3,
        ge=2,
        description="Generative A/B samples per question used to find flip-floppers.",
    )

    scoring_runs: int = Field(
        default=5,
        ge=1,
        description="Generative A/B samples per flip-flopper question used for scoring.",
    )

    temperature: float | None = Field(
        default=None,
        gt=0,
        description="Sampling temperature. None means: use the model's own "
        "generation temperature when defined, else 0.6.",
    )

    top_p: float = Field(
        default=0.95,
        gt=0,
        le=1,
        description="Top-p filter applied (before temperature) on the A/B choice.",
    )

    min_p: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Min-p filter applied (before temperature) on the A/B choice.",
    )

    seed_salt: int = Field(
        default=0,
        description="Added to the seed when deriving per-(run, question) RNG seeds.",
    )


class CalibratedPIQA(Scorer):
    """
    Scores candidates by total correct answers over repeated generative A/B
    samples on the frozen set of base-model PIQA flip-floppers.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def _resolve_temperature(self, ctx: Context) -> float:
        if self.settings.temperature is not None:
            return float(self.settings.temperature)
        model = ctx.get_model()
        generation_config = getattr(
            getattr(model.model, "generation_config", None), "temperature", None
        )
        # generation_config.temperature is None only when the model's
        # generation config leaves it unset; a plain float (usually the
        # transformers default of 1.0) means the model declares its own.
        if generation_config is None:
            return 0.6
        if generation_config <= 0:
            raise ValueError(
                f"Model generation temperature must be positive, got {generation_config!r}"
            )
        return float(generation_config)

    def _render_prompt(self, model, prompt: Prompt) -> str:
        """Render exactly what generate() would feed the model, minus the
        response prefix (managed separately)."""
        messages = model._apply_system_role(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )
        kwargs: dict = {"add_generation_prompt": True, "tokenize": False}
        # strftime_now keeps date-rendering templates reproducible (same as generate()).
        try:
            return model.tokenizer.apply_chat_template(
                messages, strftime_now=strftime_fixed_date, **kwargs
            )
        except TypeError:
            # Older tokenizer implementations without the strftime_now hook.
            return model.tokenizer.apply_chat_template(messages, **kwargs)

    def _resolve_letter_ids(
        self, model, rendered_prompts: list[str]
    ) -> tuple[list[int], list[int]]:
        """One (A-id, B-id) pair per prompt, found by construction: encode the
        literal evaluated string and take the single new trailing token."""
        tokenizer = model.tokenizer
        a_ids: list[int] = []
        b_ids: list[int] = []
        for index, rendered in enumerate(rendered_prompts):
            base_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            a_ext = tokenizer(rendered + "A", add_special_tokens=False)["input_ids"]
            b_ext = tokenizer(rendered + "B", add_special_tokens=False)["input_ids"]
            a_id = self._trailing_token(base_ids, a_ext, index, "A")
            b_id = self._trailing_token(base_ids, b_ext, index, "B")
            a_ids.append(a_id)
            b_ids.append(b_id)
        return a_ids, b_ids

    @staticmethod
    def _trailing_token(
        base_ids: list[int], extended_ids: list[int], index: int, letter: str
    ) -> int:
        common = 0
        while (
            common < len(base_ids)
            and common < len(extended_ids)
            and base_ids[common] == extended_ids[common]
        ):
            common += 1
        tail = extended_ids[common:]
        if len(tail) != 1:
            raise ValueError(
                f"PIQA prompt {index}: appending {letter!r} produced "
                f"{len(tail)} new tokens instead of 1; cannot score by first token. "
                "The chat template is likely emitting a thinking prefix — this "
                "scorer needs the first generated token to be the answer, so use "
                "a model whose chat template does not emit one."
            )
        return tail[0]

    def _draw_correct(
        self,
        log_prob_a: float,
        log_prob_b: float,
        expected_a: bool,
        temperature: float,
        generator: torch.Generator,
    ) -> bool:
        """One A/B sample: top-p and min-p filtering on the unscaled
        distribution, temperature last. Inputs are log-probs
        (from log_softmax); re-applying softmax recovers the distribution
        exactly (softmax is shift-invariant), and dividing those log-probs by
        T before the final softmax is exactly temperature scaling."""
        logits = torch.tensor([log_prob_a, log_prob_b], dtype=torch.float64)
        probs = F.softmax(logits, dim=-1)
        order = torch.argsort(probs, descending=True).tolist()
        # Top-p: smallest set with cumulative probability >= top_p.
        keep = [False, False]
        cumulative = 0.0
        for position in order:
            keep[position] = True
            cumulative += probs[position].item()
            if cumulative >= self.settings.top_p:
                break
        # Min-p: drop anything below min_p * top probability.
        top_prob = probs[order[0]].item()
        for position in range(2):
            if probs[position].item() < self.settings.min_p * top_prob:
                keep[position] = False
        kept = [position for position in range(2) if keep[position]]
        # NOTE: `kept` is never empty here: the top-p loop always keeps the top
        # token, and min-p can never drop it (p_top >= min_p * p_top for
        # min_p <= 1, enforced by the Settings bounds).
        final_probs = F.softmax(logits[kept] / temperature, dim=-1)
        expected_position = 0 if expected_a else 1
        if expected_position not in kept:
            return False
        slot = kept.index(expected_position)
        return torch.rand((), generator=generator).item() < final_probs[slot].item()

    def _seed_for(self, seed: int, run: int, index: int) -> torch.Generator:
        # Heretic itself accepts any int as `seed` (including negatives), but
        # torch.manual_seed requires 0 <= seed < 2**64. Modulo keeps every
        # legal heretic seed working while staying one-to-one in practice.
        generator = torch.Generator()
        generator.manual_seed(
            (seed + self.settings.seed_salt + run * 1_000_000 + index) % (2**64)
        )
        return generator

    def _sample_pass(
        self,
        pair_logits: torch.Tensor,
        expected: list[bool],
        temperature: float,
        seed: int,
        run_offset: int,
        num_runs: int,
    ) -> list[list[bool]]:
        """Sample num_runs independent A/B outcomes per question from fixed
        next-token logits. Returns [run][question]."""
        logprobs = F.log_softmax(pair_logits.double(), dim=-1)
        passes: list[list[bool]] = []
        for run in range(num_runs):
            outcomes: list[bool] = []
            for index in range(pair_logits.shape[0]):
                generator = self._seed_for(seed, run_offset + run, index)
                outcomes.append(
                    self._draw_correct(
                        logprobs[index, 0].item(),
                        logprobs[index, 1].item(),
                        expected[index],
                        temperature,
                        generator,
                    )
                )
            passes.append(outcomes)
        return passes

    def _pair_logits(
        self, ctx: Context, prompts: list[Prompt]
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Batched next-token logits restricted to the per-prompt A/B ids.
        Returns (N, 2) logits plus the id lists. Forwards run in heretic
        batch_size chunks so only one batch of full-vocabulary logits is
        materialized at a time — gathering from a single (N, vocab) matrix
        for the whole PIQA set would cost ~1GB on top of the model."""
        model = ctx.get_model()
        heretic_settings = self.heretic_settings
        previous_prefix = heretic_settings.response_prefix
        heretic_settings.response_prefix = ""
        try:
            rendered = [self._render_prompt(model, prompt) for prompt in prompts]
            a_ids, b_ids = self._resolve_letter_ids(model, rendered)
            # batch_size 0 means "auto" and is resolved to >= 1 during startup
            # before scorers run; fall back to a single batch when unset (tests).
            batch_size = heretic_settings.batch_size or len(prompts) or 1
            chunks: list[torch.Tensor] = []
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                logits = ctx.get_logits(batch)
                pairs = torch.stack(
                    [
                        logits[
                            index,
                            torch.tensor(
                                [a_ids[start + index], b_ids[start + index]],
                                dtype=torch.long,
                                device=logits.device,
                            ),
                        ]
                        for index in range(len(batch))
                    ]
                ).float()
                chunks.append(pairs)
            gathered = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 2)
        finally:
            heretic_settings.response_prefix = previous_prefix
        return gathered, a_ids, b_ids

    def _cache_path(self) -> Path:
        return Path(self.heretic_settings.study_checkpoint_dir) / CACHE_FILENAME

    def _effective_system_prompt(self) -> str:
        """System prompt actually rendered into the PIQA items: the per-dataset
        override when set, else the global one (mirrors load_prompts)."""
        spec = self.settings.dataset
        if spec.system_prompt is not None:
            return spec.system_prompt
        return self.heretic_settings.system_prompt

    def _compute_fingerprint(
        self, dataset_commit: str | None, temperature: float
    ) -> str:
        # Keys the FLIP SET only: the set depends on the model, the data, the
        # rendered prompts, the seed and the sampling distribution used during
        # discovery — but NOT on scoring_runs, which only affects the
        # baseline/candidate passes. (A coarse key here would force a full
        # rediscovery every time someone retunes the scoring budget.) Every
        # field that changes the prompts or the logits must appear here;
        # otherwise a stale flip set is silently reused.
        import hashlib

        spec = self.settings.dataset
        payload = {
            "model": self.heretic_settings.model,
            "model_commit": self.heretic_settings.model_commit,
            "quantization": str(self.heretic_settings.quantization),
            "dtypes": list(self.heretic_settings.dtypes),
            "dataset": spec.dataset,
            "dataset_commit": dataset_commit,
            "dataset_config": spec.config,
            "dataset_split": spec.split,
            "system_prompt": self._effective_system_prompt(),
            "prefix": spec.prefix,
            "suffix": spec.suffix,
            "seed": self.heretic_settings.seed,
            "seed_salt": self.settings.seed_salt,
            "discovery_runs": self.settings.discovery_runs,
            "temperature": temperature,
            "top_p": self.settings.top_p,
            "min_p": self.settings.min_p,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _load_cache(self) -> dict:
        """Whole cache file: {fingerprint: record}. A single file must serve
        any number of scorer instances sharing the study dir, so records are
        keyed by fingerprint instead of one instance overwriting another's."""
        try:
            record = json.loads(self._cache_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def _save_cache(self, fingerprint: str, record: dict) -> None:
        # Read-modify-write: another scorer instance may have stored its own
        # fingerprint record since this process loaded the file.
        merged = self._load_cache()
        merged[fingerprint] = record
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    @staticmethod
    def _valid_flip_ids(flip_ids: Any, num_questions: int) -> bool:
        return (
            isinstance(flip_ids, list)
            and len(flip_ids) > 0
            and all(
                isinstance(i, int)
                and not isinstance(i, bool)
                and 0 <= i < num_questions
                for i in flip_ids
            )
        )

    @staticmethod
    def _valid_base_total(base_total: Any, num_flip: int, scoring_runs: int) -> bool:
        return (
            isinstance(base_total, int)
            and not isinstance(base_total, bool)
            and 0 <= base_total <= scoring_runs * num_flip
        )

    @staticmethod
    def _piqa_user_text(goal: str, sol1: str, sol2: str) -> str:
        return (
            f"Goal: {goal}\nA. {sol1}\nB. {sol2}\n"
            "Which solution is correct? Reply with only the letter A or B."
        )

    def init(self, ctx: Context) -> None:
        seed = self.heretic_settings.seed
        if seed is None:
            raise ValueError(
                "CalibratedPIQA requires a seed (heretic `seed` setting); "
                "every draw derives from it."
            )

        print()
        print(
            f"Loading {self.settings.score_name} PIQA items "
            f"[bold]{self.settings.dataset.dataset}[/]..."
        )
        import huggingface_hub
        from datasets import load_dataset

        spec = self.settings.dataset
        if spec.commit is None:
            try:
                spec.commit = huggingface_hub.dataset_info(spec.dataset).sha
            except Exception as error:
                print(
                    "[yellow]Warning: Could not fetch the latest commit hash for dataset "
                    f"[bold]{spec.dataset}[/] ({error}). The dataset version will not be pinned.[/]"
                )
        rows = load_dataset(
            spec.dataset,
            name=spec.config,
            split=spec.split,
            revision=spec.commit,
        )
        goals = list(rows["goal"])
        sol1 = list(rows["sol1"])
        sol2 = list(rows["sol2"])
        labels = [int(label) for label in rows["label"]]
        print(f"* [bold]{len(goals)}[/] PIQA questions loaded")
        if not goals:
            raise ValueError(
                "CalibratedPIQA loaded no PIQA questions; check the dataset split."
            )

        # Honor prefix/suffix and the per-dataset system prompt override,
        # exactly as load_prompts does for other scorers.
        system = self._effective_system_prompt()
        all_prompts = [
            Prompt(
                system=system,
                user=(f"{spec.prefix} {base}" if spec.prefix else base)
                + (f" {spec.suffix}" if spec.suffix else ""),
            )
            for base in (
                self._piqa_user_text(goal, first, second)
                for goal, first, second in zip(goals, sol1, sol2)
            )
        ]
        expected_all = [label == 0 for label in labels]

        temperature = self._resolve_temperature(ctx)
        print(f"* Sampling temperature: [bold]{temperature}[/]")
        fingerprint = self._compute_fingerprint(spec.commit, temperature)

        cache_path = self._cache_path()
        record = self._load_cache().get(fingerprint, {})
        flip_ids: list[int] | None = None
        base_total: int | None = None
        if self._valid_flip_ids(record.get("flip_ids"), len(all_prompts)):
            flip_ids = sorted(record["flip_ids"])
            # The flip set does not depend on scoring_runs, but the baseline
            # total does — only reuse it when it was measured with the same
            # run count, otherwise re-run the baseline on the cached set.
            if record.get("scoring_runs") == self.settings.scoring_runs and (
                self._valid_base_total(
                    record.get("base_total"), len(flip_ids), self.settings.scoring_runs
                )
            ):
                base_total = record["base_total"]
                print(
                    f"* Resumed calibrated set from [bold]{cache_path}[/]: "
                    f"[bold]{len(flip_ids)}[/] flip-floppers, "
                    f"base [bold]{base_total}[/]"
                )

        if flip_ids is None:
            print(
                f"* Discovery: {self.settings.discovery_runs} sampling passes over "
                f"[bold]{len(all_prompts)}[/] questions..."
            )
            pair_logits, _, _ = self._pair_logits(ctx, all_prompts)
            # Forward passes are deterministic for fixed weights and inputs, so
            # the logits are computed once and sampled repeatedly.
            passes = self._sample_pass(
                pair_logits,
                expected_all,
                temperature,
                seed,
                run_offset=0,
                num_runs=self.settings.discovery_runs,
            )
            flip_ids = [
                index
                for index in range(len(all_prompts))
                if len({passes[run][index] for run in range(len(passes))}) > 1
            ]
            print(
                f"* Flip-floppers: [bold]{len(flip_ids)}[/]/{len(all_prompts)} "
                f"({100.0 * len(flip_ids) / len(all_prompts):.1f}%)"
            )
            if not flip_ids:
                raise ValueError(
                    "CalibratedPIQA found no flip-flopping PIQA questions; "
                    "the model/temperature combination is fully deterministic. "
                    "Cannot rank candidates on an empty set."
                )

        self._flip_ids = sorted(flip_ids)
        self._flip_prompts = [all_prompts[index] for index in self._flip_ids]
        self._flip_expected = [expected_all[index] for index in self._flip_ids]
        self._fingerprint_hex = fingerprint
        self._temperature = temperature

        if base_total is None:
            print(
                f"* Baseline: {self.settings.scoring_runs} scoring passes over "
                f"[bold]{len(self._flip_ids)}[/] flip-floppers..."
            )
            pair_logits, _, _ = self._pair_logits(ctx, self._flip_prompts)
            passes = self._sample_pass(
                pair_logits,
                self._flip_expected,
                temperature,
                seed,
                run_offset=SCORING_RUN_OFFSET,
                num_runs=self.settings.scoring_runs,
            )
            base_total = sum(sum(run) for run in passes)
            self._save_cache(
                fingerprint,
                {
                    "flip_ids": self._flip_ids,
                    "scoring_runs": self.settings.scoring_runs,
                    "base_total": base_total,
                },
            )
        self._base_total = base_total
        print(
            f"* Calibrated PIQA base score: "
            f"[bold]{base_total}[/]/{self.settings.scoring_runs * len(self._flip_ids)}"
        )

    def _require_state(self) -> None:
        if (
            not hasattr(self, "_flip_ids")
            or not hasattr(self, "_flip_prompts")
            or not hasattr(self, "_flip_expected")
            or not hasattr(self, "_base_total")
            or not hasattr(self, "_temperature")
        ):
            raise RuntimeError(
                "CalibratedPIQA.init() must run before scoring; "
                "the flip-flopper set is not loaded."
            )

    def _total_for(self, ctx: Context) -> tuple[int, int]:
        """Score the current model on the flip set. Returns (total, denominator)."""
        self._require_state()
        seed = self.heretic_settings.seed
        if seed is None:
            # init() already fail-fasts on this; the settings cannot change
            # mid-study, so reaching here means a programming error, not input.
            raise RuntimeError("CalibratedPIQA has no seed; init() did not run.")
        pair_logits, _, _ = self._pair_logits(ctx, self._flip_prompts)
        passes = self._sample_pass(
            pair_logits,
            self._flip_expected,
            self._temperature,
            seed,
            run_offset=SCORING_RUN_OFFSET,
            num_runs=self.settings.scoring_runs,
        )
        total = sum(sum(run) for run in passes)
        return total, self.settings.scoring_runs * len(self._flip_ids)

    def get_score(self, ctx: Context) -> Score:
        total, denominator = self._total_for(ctx)
        base_total = self._base_total
        delta = total - base_total
        value = total / denominator
        print(
            f"* Calibrated PIQA: [bold]{total}[/]/{denominator} "
            f"(base [bold]{base_total}[/], Δ {delta:+d})"
        )
        return Score(
            value=value,
            rich_display=f"[bold]{total}[/]/{denominator} "
            f"(base {base_total}, Δ {delta:+d})",
            md_display=f"{total}/{denominator} (base {base_total}, Δ {delta:+d})",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        self._require_state()
        denominator = self.settings.scoring_runs * len(self._flip_ids)
        return Score(
            value=self._base_total / denominator,
            rich_display=f"[bold]{self._base_total}[/]/{denominator}",
            md_display=f"{self._base_total}/{denominator}",
        )
