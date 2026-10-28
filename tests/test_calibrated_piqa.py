# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import io
import json
import tempfile
import types
import unittest
from typing import Any, cast
from unittest.mock import patch

import torch
from datasets import Dataset
from pydantic import ValidationError
from rich.console import Console

from heretic.config import Settings as HereticSettings
from heretic.scorer import Context
from heretic.scorers.calibrated_piqa import CalibratedPIQA, Settings
from heretic.utils import Prompt, is_dataset_specification_reproducible


class FakeTokenizer:
    """Character-level tokenizer: each character maps to its ord().

    Appending "A"/"B" therefore always yields exactly one new trailing token,
    mirroring the happy path of id resolution.
    """

    def __init__(self, fail_on_strftime: bool = False) -> None:
        self.fail_on_strftime = fail_on_strftime
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(dict(kwargs))
        if self.fail_on_strftime and "strftime_now" in kwargs:
            raise TypeError("no strftime_now hook")
        return (
            "\n".join(f"{m['role']}:{m['content']}" for m in messages) + "\nassistant:"
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class FakeModel:
    def __init__(
        self,
        tokenizer=None,
        logits: torch.Tensor | None = None,
        gen_temp: Any = None,
        fail_logits: bool = False,
    ) -> None:
        self.tokenizer = tokenizer or FakeTokenizer()
        if gen_temp == "absent":
            self.model = types.SimpleNamespace()
        else:
            self.model = types.SimpleNamespace(
                generation_config=types.SimpleNamespace(temperature=gen_temp)
            )
        self._logits = logits
        self.logits_calls = 0
        self.fail_logits = fail_logits
        self.seen_prompts: list[list[Prompt]] = []

    def _apply_system_role(self, messages):
        return messages

    def get_logits_batched(self, prompts: list[Prompt]) -> torch.Tensor:
        self.logits_calls += 1
        self.seen_prompts.append(prompts)
        if self.fail_logits:
            raise RuntimeError("forward failed")
        assert self._logits is not None
        return self._logits


class FakeCtx:
    def __init__(self, model: FakeModel) -> None:
        self._model = model

    def get_model(self):
        return self._model

    def get_logits(self, prompts: list[Prompt]) -> torch.Tensor:
        # Managed API path, exactly as the plugin calls it per batch.
        return self._model.get_logits_batched(prompts)


def make_heretic_settings(**overrides: Any):
    hs = HereticSettings.model_construct()
    hs.seed = 1234
    hs.system_prompt = "SYS"
    hs.response_prefix = ""
    hs.model = "some-model"
    hs.model_commit = None
    hs.study_checkpoint_dir = "checkpoints"
    for key, value in overrides.items():
        setattr(hs, key, value)
    return hs


def make_scorer(tmpdir: str, **settings_kwargs: Any) -> CalibratedPIQA:
    hs = make_heretic_settings(study_checkpoint_dir=tmpdir)
    return CalibratedPIQA(heretic_settings=hs, settings=Settings(**settings_kwargs))


def ab_logits(pairs: list[tuple[float, float]], vocab: int = 256) -> torch.Tensor:
    """Logit matrix with the given (A, B) values at the char-level ids."""
    tensor = torch.zeros(len(pairs), vocab)
    for index, (la, lb) in enumerate(pairs):
        tensor[index, ord("A")] = la
        tensor[index, ord("B")] = lb
    return tensor


def fake_rows():
    return Dataset.from_dict(
        {
            "goal": ["g0", "g1"],
            "sol1": ["a0", "a1"],
            "sol2": ["b0", "b1"],
            "label": [0, 1],
        }
    )


def capture_print():
    output = io.StringIO()
    console = Console(file=output, highlight=False, width=200)
    return output, console


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = Settings()
        self.assertEqual(settings.score_name, "Calibrated PIQA")
        self.assertEqual(settings.dataset.dataset, "baber/piqa")
        self.assertEqual(settings.dataset.split, "validation")
        self.assertEqual(settings.discovery_runs, 3)
        self.assertEqual(settings.scoring_runs, 5)
        self.assertIsNone(settings.temperature)
        self.assertEqual(settings.top_p, 0.95)
        self.assertEqual(settings.min_p, 0.05)
        self.assertEqual(settings.seed_salt, 0)

    def test_rejects_nonpositive_temperature(self) -> None:
        for temp in [0.0, -0.6]:
            with self.subTest(temp=temp):
                with self.assertRaises(ValidationError):
                    Settings(temperature=temp)

    def test_rejects_bad_top_p(self) -> None:
        for top_p in [0.0, 1.5, -0.1]:
            with self.subTest(top_p=top_p):
                with self.assertRaises(ValidationError):
                    Settings(top_p=top_p)

    def test_rejects_bad_min_p(self) -> None:
        for min_p in [-0.1, 1.5]:
            with self.subTest(min_p=min_p):
                with self.assertRaises(ValidationError):
                    Settings(min_p=min_p)

    def test_rejects_too_few_runs(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(discovery_runs=1)
        with self.assertRaises(ValidationError):
            Settings(scoring_runs=0)

    def test_dataset_spec_is_collected_for_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            specs = scorer.get_dataset_specifications()
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].dataset, "baber/piqa")

    def test_plugin_contract(self) -> None:
        CalibratedPIQA.validate_contract()
        self.assertNotIn("__init__", CalibratedPIQA.__dict__)
        self.assertIs(CalibratedPIQA.get_settings_model(), Settings)

    def test_score_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(make_scorer(tmpdir).score_name, "Calibrated PIQA")
            custom = make_scorer(tmpdir, score_name="Custom")
            self.assertEqual(custom.score_name, "Custom")

    def test_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(make_scorer(tmpdir).reproducible)


class ResolveTemperatureTests(unittest.TestCase):
    def _scorer(self, tmpdir: str, **kwargs: Any) -> CalibratedPIQA:
        return make_scorer(tmpdir, **kwargs)

    def test_explicit_setting_wins_over_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir, temperature=0.3)
            ctx = FakeCtx(FakeModel(gen_temp=0.9))
            self.assertEqual(scorer._resolve_temperature(cast(Context, ctx)), 0.3)

    def test_model_temperature_used_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            ctx = FakeCtx(FakeModel(gen_temp=0.7))
            self.assertEqual(scorer._resolve_temperature(cast(Context, ctx)), 0.7)

    def test_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            for model in [FakeModel(gen_temp=None), FakeModel(gen_temp="absent")]:
                with self.subTest(model=model):
                    ctx = FakeCtx(model)
                    self.assertEqual(
                        scorer._resolve_temperature(cast(Context, ctx)), 0.6
                    )

    def test_rejects_nonpositive_model_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            ctx = FakeCtx(FakeModel(gen_temp=0.0))
            with self.assertRaisesRegex(ValueError, "positive"):
                scorer._resolve_temperature(cast(Context, ctx))


class TrailingTokenTests(unittest.TestCase):
    def test_single_new_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            self.assertEqual(scorer._trailing_token([1, 2, 3], [1, 2, 3, 4], 0, "A"), 4)

    def test_merged_token_still_single(self) -> None:
        # Boundary merge: "ab" + "A" tokenizes as [..., "abA"].
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            self.assertEqual(scorer._trailing_token([1, 2], [1, 9], 5, "B"), 9)

    def test_multiple_new_tokens_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with self.assertRaisesRegex(ValueError, "instead of 1"):
                scorer._trailing_token([1, 2, 3], [1, 2, 4, 5], 7, "A")


class RenderAndResolveTests(unittest.TestCase):
    def test_render_contains_system_and_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            model = FakeModel()
            rendered = scorer._render_prompt(
                model, Prompt(system="SYS", user="the question")
            )
            self.assertIn("SYS", rendered)
            self.assertIn("the question", rendered)

    def test_strftime_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            model = FakeModel(tokenizer=FakeTokenizer(fail_on_strftime=True))
            rendered = scorer._render_prompt(
                model, Prompt(system="SYS", user="the question")
            )
            self.assertIn("the question", rendered)
            # First attempt carried strftime_now and failed; retry did not.
            self.assertIn("strftime_now", model.tokenizer.template_calls[0])
            self.assertNotIn("strftime_now", model.tokenizer.template_calls[1])

    def test_resolve_letter_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            model = FakeModel()
            a_ids, b_ids = scorer._resolve_letter_ids(
                model, ["first prompt", "second prompt"]
            )
            self.assertEqual(a_ids, [ord("A"), ord("A")])
            self.assertEqual(b_ids, [ord("B"), ord("B")])


class DrawTests(unittest.TestCase):
    def _scorer(self, tmpdir: str, **kwargs: Any) -> CalibratedPIQA:
        return make_scorer(tmpdir, **kwargs)

    def test_top_p_can_drop_bottom_token(self) -> None:
        # p_top ~= 1.0 >= 0.95: bottom is dropped, so expecting B is always wrong
        # and expecting A is always right (u < 1.0 always holds).
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            for seed in range(5):
                gen = scorer._seed_for(99, 1000, seed)
                self.assertFalse(scorer._draw_correct(10.0, -10.0, False, 0.6, gen))
                gen = scorer._seed_for(99, 1000, seed)
                self.assertTrue(scorer._draw_correct(10.0, -10.0, True, 0.6, gen))

    def test_min_p_can_drop_bottom_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir, min_p=0.05)
            gen = scorer._seed_for(1, 1000, 0)
            # p_B ~= 4.5e-5 < 0.05 * p_A: dropped.
            self.assertFalse(scorer._draw_correct(0.0, -10.0, False, 0.6, gen))

    def test_disabled_filters_keep_both_tokens(self) -> None:
        # Skewed enough that the default filters drop B (deterministically
        # False), but mild enough to flip sometimes when filtering is off.
        with tempfile.TemporaryDirectory() as tmpdir:
            disabled = self._scorer(tmpdir, top_p=1.0, min_p=0.0)
            outcomes = {
                disabled._draw_correct(
                    0.0, -3.0, False, 1.0, disabled._seed_for(7, i, 0)
                )
                for i in range(200)
            }
            self.assertEqual(outcomes, {True, False})
            enabled = self._scorer(tmpdir)
            self.assertFalse(
                any(
                    enabled._draw_correct(
                        0.0, -3.0, False, 1.0, enabled._seed_for(7, i, 0)
                    )
                    for i in range(200)
                )
            )

    def test_uniform_comparison_both_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            gen = scorer._seed_for(1, 1000, 0)
            with patch("torch.rand", return_value=torch.tensor(0.99)):
                self.assertFalse(scorer._draw_correct(0.0, 0.0, True, 1.0, gen))
            gen = scorer._seed_for(1, 1000, 0)
            with patch("torch.rand", return_value=torch.tensor(0.01)):
                self.assertTrue(scorer._draw_correct(0.0, 0.0, True, 1.0, gen))

    def test_temperature_is_applied(self) -> None:
        # At T=1 these logits give p_A ~= 0.525 (a coin flip); at T=0.01 the
        # outcome is essentially deterministic. All-True across 20 seeds proves
        # the temperature division happens (a T-ignoring sampler would fail).
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            for seed in range(20):
                gen = scorer._seed_for(7, 1000, seed)
                self.assertTrue(scorer._draw_correct(0.1, 0.0, True, 0.01, gen))

    def test_same_seed_same_draw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            first = scorer._draw_correct(
                0.3, 0.0, True, 0.6, scorer._seed_for(7, 1000, 3)
            )
            second = scorer._draw_correct(
                0.3, 0.0, True, 0.6, scorer._seed_for(7, 1000, 3)
            )
            self.assertEqual(first, second)

    def test_mid_probs_flip_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._scorer(tmpdir)
            outcomes = {
                scorer._draw_correct(0.0, 0.0, True, 1.0, scorer._seed_for(7, i, 0))
                for i in range(200)
            }
            self.assertEqual(outcomes, {True, False})


class SeedTests(unittest.TestCase):
    def test_seed_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            generator = scorer._seed_for(1234, 1000, 7)
            self.assertEqual(generator.initial_seed(), 1234 + 1000 * 1_000_000 + 7)
            other = scorer._seed_for(1234, 1001, 7)
            self.assertNotEqual(generator.initial_seed(), other.initial_seed())

    def test_negative_seed_works(self) -> None:
        # Heretic accepts any int as seed; torch.manual_seed does not take
        # negatives, so the derivation must map into [0, 2**64).
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            generator = scorer._seed_for(-1, 1000, 7)
            self.assertEqual(
                generator.initial_seed(), (-1 + 1000 * 1_000_000 + 7) % (2**64)
            )
            self.assertIn(
                scorer._draw_correct(0.0, 0.0, True, 1.0, generator),
                (True, False),
            )


class SamplePassTests(unittest.TestCase):
    def test_shape_and_expected_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            pairs = torch.tensor(
                [[20.0, -20.0], [-20.0, 20.0], [20.0, -20.0], [-20.0, 20.0]]
            )
            passes = scorer._sample_pass(
                pairs, [True, False, True, False], 0.6, 1234, 0, 3
            )
            self.assertEqual(len(passes), 3)
            for run in passes:
                self.assertEqual(run, [True, True, True, True])

    def test_deterministic_for_same_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            pairs = torch.tensor([[0.4, 0.0], [0.0, 0.4]])
            first = scorer._sample_pass(pairs, [True, False], 0.6, 1234, 0, 2)
            second = scorer._sample_pass(pairs, [True, False], 0.6, 1234, 0, 2)
            self.assertEqual(first, second)

    def test_run_offset_changes_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            seen: list[tuple[int, int, int]] = []
            real = scorer._seed_for

            def spy(seed: int, run: int, index: int) -> torch.Generator:
                seen.append((seed, run, index))
                return real(seed, run, index)

            with patch.object(scorer, "_seed_for", side_effect=spy):
                scorer._sample_pass(
                    torch.zeros(2, 2), [True, False], 0.6, 1234, 1000, 2
                )
            self.assertEqual(
                seen,
                [(1234, 1000, 0), (1234, 1000, 1), (1234, 1001, 0), (1234, 1001, 1)],
            )


class PairLogitsTests(unittest.TestCase):
    def _setup(self, tmpdir: str):
        scorer = make_scorer(tmpdir)
        prompts = [
            Prompt(system="SYS", user="Goal: g0\nA. a0\nB. b0\nWhich?"),
            Prompt(system="SYS", user="Goal: g1\nA. a1\nB. b1\nWhich?"),
        ]
        return scorer, prompts

    def test_gathers_ab_logits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer, prompts = self._setup(tmpdir)
            model = FakeModel(logits=ab_logits([(1.5, -0.5), (-1.0, 2.0)]))
            pairs, a_ids, b_ids = scorer._pair_logits(
                cast(Context, FakeCtx(model)), prompts
            )
            self.assertEqual(a_ids, [ord("A"), ord("A")])
            self.assertEqual(b_ids, [ord("B"), ord("B")])
            self.assertTrue(
                torch.equal(pairs, torch.tensor([[1.5, -0.5], [-1.0, 2.0]]))
            )
            self.assertEqual(model.logits_calls, 1)
            # Prompts reach the model with identical content (batched slices
            # are new list objects, so identity cannot hold).
            self.assertEqual(model.seen_prompts[0], prompts)

    def test_response_prefix_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer.heretic_settings.response_prefix = "PREFIX"
            model = FakeModel(logits=ab_logits([(0.0, 0.0)]))
            scorer._pair_logits(
                cast(Context, FakeCtx(model)),
                [Prompt(system="SYS", user="q")],
            )
            self.assertEqual(scorer.heretic_settings.response_prefix, "PREFIX")

    def test_response_prefix_restored_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer.heretic_settings.response_prefix = "PREFIX"
            model = FakeModel(fail_logits=True)
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                scorer._pair_logits(
                    cast(Context, FakeCtx(model)),
                    [Prompt(system="SYS", user="q")],
                )
            self.assertEqual(scorer.heretic_settings.response_prefix, "PREFIX")

    def test_forwards_run_without_response_prefix(self) -> None:
        # Logits must be measured at the bare chat template, not after a
        # leftover response prefix from candidate generation.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer.heretic_settings.response_prefix = "PREFIX"
            model = FakeModel(logits=ab_logits([(0.0, 0.0)]))
            seen: list = []
            real = model.get_logits_batched

            def rec(batch: list[Prompt]) -> torch.Tensor:
                seen.append(scorer.heretic_settings.response_prefix)
                return real(batch)

            model.get_logits_batched = rec  # type: ignore[method-assign]
            scorer._pair_logits(
                cast(Context, FakeCtx(model)),
                [Prompt(system="SYS", user="q")],
            )
            self.assertEqual(seen, [""])
            self.assertEqual(scorer.heretic_settings.response_prefix, "PREFIX")

    def test_batches_forward_passes(self) -> None:
        # batch_size=1 over two prompts: two forward calls, and the gathered
        # pairs still align with the global prompt order.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer, prompts = self._setup(tmpdir)
            scorer.heretic_settings.batch_size = 1
            model = FakeModel()
            per_batch = [ab_logits([(1.5, -0.5)]), ab_logits([(-1.0, 2.0)])]

            def fake_batched(batch: list[Prompt]) -> torch.Tensor:
                model.seen_prompts.append(batch)
                model.logits_calls += 1
                return per_batch[model.logits_calls - 1]

            model.get_logits_batched = fake_batched  # type: ignore[method-assign]
            pairs, a_ids, b_ids = scorer._pair_logits(
                cast(Context, FakeCtx(model)), prompts
            )
            self.assertEqual(model.logits_calls, 2)
            self.assertEqual([len(b) for b in model.seen_prompts], [1, 1])
            self.assertEqual(a_ids, [ord("A"), ord("A")])
            self.assertEqual(b_ids, [ord("B"), ord("B")])
            self.assertTrue(
                torch.equal(pairs, torch.tensor([[1.5, -0.5], [-1.0, 2.0]]))
            )


class FingerprintTests(unittest.TestCase):
    def test_stable_and_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            first = scorer._compute_fingerprint("abc123", 0.6)
            self.assertEqual(first, scorer._compute_fingerprint("abc123", 0.6))
            self.assertEqual(len(first), 64)
            self.assertNotEqual(first, scorer._compute_fingerprint("abc123", 0.7))
            scorer.heretic_settings.seed = 9999
            self.assertNotEqual(first, scorer._compute_fingerprint("abc123", 0.6))

    def test_sensitive_to_prompt_and_precision_keys(self) -> None:
        # Every field that changes the rendered prompts or the logits must
        # change the fingerprint; otherwise a stale flip set is reused.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            base = scorer._compute_fingerprint("abc123", 0.6)
            scorer.heretic_settings.system_prompt = "OTHER"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.heretic_settings.system_prompt = "SYS"

            scorer.settings.dataset.system_prompt = "OVERRIDE"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.dataset.system_prompt = None

            scorer.settings.dataset.prefix = "PRE"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.dataset.prefix = ""

            scorer.settings.dataset.config = "plain_text"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.dataset.config = None

            scorer.settings.dataset.split = "train"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.dataset.split = "validation"

            scorer.settings.dataset.suffix = "POST"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.dataset.suffix = ""

            scorer.settings.top_p = 0.5
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.top_p = 0.95

            scorer.settings.seed_salt = 7
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.seed_salt = 0

            scorer.settings.discovery_runs = 5
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))
            scorer.settings.discovery_runs = 3

            scorer.heretic_settings.quantization = "bnb_4bit"
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))

            scorer.heretic_settings.dtypes = ["float16"]
            self.assertNotEqual(base, scorer._compute_fingerprint("abc123", 0.6))


class CacheTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            record = {
                "flip_ids": [3, 7],
                "scoring_runs": 5,
                "base_total": 31,
            }
            scorer._save_cache("fp", record)
            self.assertTrue(scorer._cache_path().is_file())
            self.assertEqual(scorer._load_cache(), {"fp": record})

    def test_merge_keeps_other_fingerprints(self) -> None:
        # Two scorer instances (e.g. different settings) share one file.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer._save_cache("fp-a", {"flip_ids": [1]})
            scorer._save_cache("fp-b", {"flip_ids": [2]})
            cached = scorer._load_cache()
            self.assertEqual(cached["fp-a"], {"flip_ids": [1]})
            self.assertEqual(cached["fp-b"], {"flip_ids": [2]})

    def test_missing_file_gives_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            self.assertEqual(scorer._load_cache(), {})

    def test_non_dict_file_gives_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer._cache_path().write_text("[1, 2, 3]", encoding="utf-8")
            self.assertEqual(scorer._load_cache(), {})


class InitTests(unittest.TestCase):
    def _run_init(self, tmpdir: str, draw_script, pair_behavior="scripted"):
        """Run init() with mocked dataset loading and hub pinning.

        draw_script: side_effect list for _draw_correct (discovery run-major,
        then baseline), or None to use the real sampler.
        pair_behavior: "scripted" patches _pair_logits to zeros and returns
        the mock; "real" uses FakeModel logits supplied via pair_logits.
        """
        scorer = make_scorer(tmpdir)
        model = FakeModel()
        ctx = FakeCtx(model)
        mocks = {}
        hub_patch = patch(
            "huggingface_hub.dataset_info",
            return_value=types.SimpleNamespace(sha="abc123"),
        )
        ds_patch = patch("datasets.load_dataset", return_value=fake_rows())
        mocks["hub"] = hub_patch.start()
        mocks["ds"] = ds_patch.start()
        self.addCleanup(hub_patch.stop)
        self.addCleanup(ds_patch.stop)
        pair_patch = None
        if pair_behavior == "scripted":
            # Size the dummy logits to the actual prompt batch: discovery runs
            # on all questions, the baseline only on the flip set.
            def fake_pair(ctx, prompts):
                count = len(prompts)
                return (
                    torch.zeros(count, 2),
                    [65] * count,
                    [66] * count,
                )

            pair_patch = patch.object(scorer, "_pair_logits", side_effect=fake_pair)
            mocks["pair"] = pair_patch.start()
            self.addCleanup(pair_patch.stop)
        draw_patch = None
        if draw_script is not None:
            draw_patch = patch.object(
                scorer, "_draw_correct", side_effect=list(draw_script)
            )
            mocks["draw"] = draw_patch.start()
            self.addCleanup(draw_patch.stop)
        output, console = capture_print()
        with patch("heretic.scorers.calibrated_piqa.print", console.print):
            scorer.init(cast(Context, ctx))
        return scorer, model, mocks, output.getvalue()

    def test_full_flow(self) -> None:
        # Discovery (run-major): Q0 [T,T,F] flips, Q1 [T,T,T] does not.
        # Baseline on Q0: [T,F,T,T,F] -> base_total 3.
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer, _, _, text = self._run_init(tmpdir, script)
            self.assertEqual(scorer._flip_ids, [0])
            self.assertEqual(scorer._flip_expected, [True])
            self.assertEqual(len(scorer._flip_prompts), 1)
            self.assertIn("Goal: g0", scorer._flip_prompts[0].user)
            self.assertEqual(scorer._base_total, 3)
            self.assertEqual(scorer._temperature, 0.6)
            self.assertIn("Flip-floppers: 1/2", text)
            self.assertIn("3/5", text)
            # Dataset commit pinned from the hub.
            self.assertEqual(scorer.settings.dataset.commit, "abc123")
            # Cache record persisted for resume, keyed by fingerprint.
            stored = json.loads(scorer._cache_path().read_text(encoding="utf-8"))
            record = stored[scorer._fingerprint_hex]
            self.assertEqual(record["flip_ids"], [0])
            self.assertEqual(record["base_total"], 3)
            self.assertEqual(record["scoring_runs"], 5)

    def test_prefix_suffix_and_system_override_applied(self) -> None:
        # Mirrors load_prompts: prefix/suffix wrap the user text and the
        # per-dataset system prompt wins over the global one.
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer.settings.dataset.prefix = "PRE"
            scorer.settings.dataset.suffix = "POST"
            scorer.settings.dataset.system_prompt = "OVERRIDE"
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    return_value=types.SimpleNamespace(sha="abc123"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(scorer, "_draw_correct", side_effect=list(script)),
            ):
                _, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(len(scorer._flip_prompts), 1)
            prompt = scorer._flip_prompts[0]
            self.assertEqual(prompt.system, "OVERRIDE")
            self.assertTrue(prompt.user.startswith("PRE Goal: g0"))
            self.assertTrue(prompt.user.endswith("POST"))

    def test_empty_dataset_fails_fast(self) -> None:
        empty = Dataset.from_dict({"goal": [], "sol1": [], "sol2": [], "label": []})
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    return_value=types.SimpleNamespace(sha="abc123"),
                ),
                patch("datasets.load_dataset", return_value=empty),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=AssertionError("must not forward on empty data"),
                ),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                with self.assertRaisesRegex(ValueError, "no PIQA questions"):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))

    def test_empty_flip_set_fails_fast(self) -> None:
        # Extreme matching logits: every draw correct -> nothing flips.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            model = FakeModel(logits=ab_logits([(20.0, -20.0), (-20.0, 20.0)]))
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    return_value=types.SimpleNamespace(sha="abc123"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                with self.assertRaisesRegex(ValueError, "no flip-flopping"):
                    scorer.init(cast(Context, FakeCtx(model)))

    def test_seed_none_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            scorer.heretic_settings.seed = None
            with (
                patch(
                    "datasets.load_dataset",
                    side_effect=AssertionError("must not load"),
                ),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                with self.assertRaisesRegex(ValueError, "seed"):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))

    def test_offline_pin_warns_and_continues(self) -> None:
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            model = FakeModel()
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    side_effect=RuntimeError("offline"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()) as mock_ds,
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(scorer, "_draw_correct", side_effect=list(script)),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(model)))
            self.assertIn("will not be pinned", output.getvalue())
            self.assertIsNone(scorer.settings.dataset.commit)
            _, kwargs = mock_ds.call_args
            self.assertIsNone(kwargs["revision"])
            self.assertEqual(scorer._base_total, 3)

    def test_resume_full_skips_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            fingerprint = scorer._compute_fingerprint("abc123", 0.6)
            scorer.settings.dataset.commit = "abc123"
            scorer._save_cache(
                fingerprint,
                {"flip_ids": [1], "scoring_runs": 5, "base_total": 4},
            )
            with (
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=AssertionError("must not be called"),
                ),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(scorer._flip_ids, [1])
            self.assertEqual(scorer._base_total, 4)
            self.assertIn("Resumed", output.getvalue())

    def test_resume_baseline_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            fingerprint = scorer._compute_fingerprint("abc123", 0.6)
            scorer.settings.dataset.commit = "abc123"
            scorer._save_cache(fingerprint, {"flip_ids": [0]})
            calls: list[int] = []

            def spy(ctx, prompts):
                calls.append(len(prompts))
                count = len(prompts)
                return torch.zeros(count, 2), [65] * count, [66] * count

            with (
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(scorer, "_pair_logits", side_effect=spy),
                patch.object(
                    scorer,
                    "_draw_correct",
                    side_effect=[True, False, True, True, False],
                ),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))
            # Discovery skipped (one forward batch: the baseline only).
            self.assertEqual(calls, [1])
            self.assertEqual(scorer._base_total, 3)
            self.assertIn("3/5", output.getvalue())
            stored = json.loads(scorer._cache_path().read_text(encoding="utf-8"))
            self.assertEqual(stored[scorer._fingerprint_hex]["base_total"], 3)

    def test_scoring_runs_change_reuses_flip_set(self) -> None:
        # The flip set does not depend on scoring_runs: same fingerprint must
        # skip discovery and only re-run the baseline with the new run count.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir, scoring_runs=5)
            fingerprint = scorer._compute_fingerprint("abc123", 0.6)
            scorer.settings.dataset.commit = "abc123"
            scorer._save_cache(
                fingerprint, {"flip_ids": [0], "scoring_runs": 2, "base_total": 2}
            )
            calls: list[int] = []

            def spy(ctx, prompts):
                calls.append(len(prompts))
                count = len(prompts)
                return torch.zeros(count, 2), [65] * count, [66] * count

            with (
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(scorer, "_pair_logits", side_effect=spy),
                patch.object(
                    scorer,
                    "_draw_correct",
                    side_effect=[True, False, True, True, False],
                ),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))
            # Discovery skipped (one forward batch: the baseline only).
            self.assertEqual(calls, [1])
            self.assertEqual(scorer._flip_ids, [0])
            self.assertEqual(scorer._base_total, 3)
            self.assertIn("3/5", output.getvalue())
            stored = json.loads(scorer._cache_path().read_text(encoding="utf-8"))
            self.assertEqual(
                stored[fingerprint],
                {"flip_ids": [0], "scoring_runs": 5, "base_total": 3},
            )

    def test_corrupt_cache_entry_triggers_recompute(self) -> None:
        # A schema-valid JSON file with a garbage record must be treated as a
        # miss (clean rediscovery), never crash scoring.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            fingerprint = scorer._compute_fingerprint("abc123", 0.6)
            scorer.settings.dataset.commit = "abc123"
            scorer._save_cache(fingerprint, {"flip_ids": ["not-an-id"]})
            script = [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
            ]
            with (
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(scorer, "_draw_correct", side_effect=list(script)),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    scorer.init(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(scorer._flip_ids, [0])
            self.assertEqual(scorer._base_total, 3)
            # Proves rediscovery ran instead of trusting the garbage record.
            self.assertIn("Flip-floppers: 1/2", output.getvalue())


class ReproducibilityTests(unittest.TestCase):
    """Coverage for the properties the upload reproducibility gate relies on
    (main.py): all scorer specs reproducible, scorer reproducible, and a
    calibration that restarts bit-identically."""

    def test_pinned_specs_pass_upload_gate(self) -> None:
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    return_value=types.SimpleNamespace(sha="abc123"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(scorer, "_draw_correct", side_effect=list(script)),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                scorer.init(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(scorer.settings.dataset.commit, "abc123")
            self.assertTrue(
                is_dataset_specification_reproducible(
                    scorer.get_dataset_specifications()
                )
            )

    def test_unpinned_specs_fail_upload_gate(self) -> None:
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    side_effect=RuntimeError("offline"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    scorer,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(scorer, "_draw_correct", side_effect=list(script)),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                scorer.init(cast(Context, FakeCtx(FakeModel())))
            self.assertIsNone(scorer.settings.dataset.commit)
            self.assertFalse(
                is_dataset_specification_reproducible(
                    scorer.get_dataset_specifications()
                )
            )

    def test_calibration_identical_across_dirs(self) -> None:
        # Real sampler, no scripts: two independent calibrations with the same
        # seed must agree bit-for-bit (no hidden nondeterminism anywhere in
        # the discovery/baseline chain).
        calibrations = []
        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
        ):
            for tmpdir in (dir1, dir2):
                scorer = make_scorer(tmpdir)
                model = FakeModel(logits=ab_logits([(0.0, 0.0), (0.0, 0.0)]))
                with (
                    patch(
                        "huggingface_hub.dataset_info",
                        return_value=types.SimpleNamespace(sha="abc123"),
                    ),
                    patch("datasets.load_dataset", return_value=fake_rows()),
                    patch("heretic.scorers.calibrated_piqa.print"),
                ):
                    scorer.init(cast(Context, FakeCtx(model)))
                calibrations.append(
                    (scorer._flip_ids, scorer._base_total, scorer._temperature)
                )
        self.assertEqual(calibrations[0], calibrations[1])
        self.assertTrue(len(calibrations[0][0]) > 0)

    def test_resume_across_instances(self) -> None:
        # A restart means a fresh scorer object: it must reproduce the exact
        # calibration from cache without any forward passes.
        script = [True, True, True, True, False, True, True, False, True, True, False]
        with tempfile.TemporaryDirectory() as tmpdir:
            first = make_scorer(tmpdir)
            with (
                patch(
                    "huggingface_hub.dataset_info",
                    return_value=types.SimpleNamespace(sha="abc123"),
                ),
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    first,
                    "_pair_logits",
                    side_effect=lambda ctx, prompts: (
                        torch.zeros(len(prompts), 2),
                        [65] * len(prompts),
                        [66] * len(prompts),
                    ),
                ),
                patch.object(first, "_draw_correct", side_effect=list(script)),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                first.init(cast(Context, FakeCtx(FakeModel())))

            second = make_scorer(tmpdir)
            # Production re-pins the same commit via the hub on restart;
            # set it directly since the hub is not mocked for this instance.
            second.settings.dataset.commit = "abc123"
            with (
                patch("datasets.load_dataset", return_value=fake_rows()),
                patch.object(
                    second,
                    "_pair_logits",
                    side_effect=AssertionError("must resume, not recompute"),
                ),
                patch("heretic.scorers.calibrated_piqa.print"),
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    second.init(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(second._fingerprint_hex, first._fingerprint_hex)
            self.assertEqual(second._flip_ids, first._flip_ids)
            self.assertEqual(second._flip_expected, first._flip_expected)
            self.assertEqual(second._base_total, first._base_total)
            self.assertIn("Resumed", output.getvalue())


class GetScoreTests(unittest.TestCase):
    def _ready_scorer(self, tmpdir: str, **kwargs: Any) -> CalibratedPIQA:
        scorer = make_scorer(tmpdir, **kwargs)
        scorer._flip_ids = [0, 1]
        scorer._flip_prompts = [
            Prompt(system="SYS", user="q0"),
            Prompt(system="SYS", user="q1"),
        ]
        scorer._flip_expected = [True, False]
        scorer._base_total = 3
        scorer._temperature = 0.6
        return scorer

    def test_score_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._ready_scorer(tmpdir, scoring_runs=2)
            pairs = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
            with patch.object(
                scorer, "_pair_logits", return_value=(pairs, [65, 66], [65, 66])
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    first = scorer.get_score(cast(Context, FakeCtx(FakeModel())))
                    second = scorer.get_score(cast(Context, FakeCtx(FakeModel())))
            # Both questions always correct over 2 runs: 4/4, base 3 -> +1.
            self.assertEqual(first.value, 1.0)
            self.assertEqual(first.value, second.value)
            self.assertIn("4/4", first.md_display)
            self.assertIn("Δ +1", first.rich_display)
            self.assertIn("Δ +1", first.md_display)
            self.assertIn("4/4", output.getvalue())

    def test_negative_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._ready_scorer(tmpdir, scoring_runs=2)
            scorer._base_total = 4
            pairs = torch.tensor([[-20.0, 20.0], [20.0, -20.0]])
            with patch.object(
                scorer, "_pair_logits", return_value=(pairs, [65, 66], [65, 66])
            ):
                output, console = capture_print()
                with patch("heretic.scorers.calibrated_piqa.print", console.print):
                    score = scorer.get_score(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(score.value, 0.0)
            self.assertIn("Δ -4", score.rich_display)
            self.assertIn("0/4", output.getvalue())

    def test_uses_scoring_stream_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = self._ready_scorer(tmpdir, scoring_runs=2)
            seen: list[tuple[int, int, int]] = []
            real = scorer._seed_for

            def spy(seed: int, run: int, index: int) -> torch.Generator:
                seen.append((seed, run, index))
                return real(seed, run, index)

            pairs = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
            with (
                patch.object(
                    scorer, "_pair_logits", return_value=(pairs, [65, 66], [65, 66])
                ),
                patch.object(scorer, "_seed_for", side_effect=spy),
            ):
                with patch("heretic.scorers.calibrated_piqa.print"):
                    scorer.get_score(cast(Context, FakeCtx(FakeModel())))
            runs = sorted({run for _, run, _ in seen})
            self.assertEqual(runs, [1000, 1001])

    def test_requires_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "init"):
                scorer.get_score(cast(Context, FakeCtx(FakeModel())))

    def test_partial_state_still_requires_init(self) -> None:
        # _require_state must cover every attribute scoring touches, not
        # just the ids — a half-initialized scorer must fail, not mis-score.
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir, scoring_runs=2)
            scorer._flip_ids = [0, 1]
            scorer._base_total = 3
            with self.assertRaisesRegex(RuntimeError, "init"):
                scorer.get_score(cast(Context, FakeCtx(FakeModel())))
            with self.assertRaisesRegex(RuntimeError, "init"):
                scorer.get_baseline_score(cast(Context, FakeCtx(FakeModel())))


class BaselineTests(unittest.TestCase):
    def test_baseline_value_and_displays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir, scoring_runs=2)
            scorer._flip_ids = [0, 1]
            scorer._flip_prompts = [
                Prompt(system="SYS", user="q0"),
                Prompt(system="SYS", user="q1"),
            ]
            scorer._flip_expected = [True, False]
            scorer._base_total = 3
            scorer._temperature = 0.6
            score = scorer.get_baseline_score(cast(Context, FakeCtx(FakeModel())))
            self.assertEqual(score.value, 0.75)
            self.assertIn("3/4", score.md_display)
            self.assertIn("3", score.rich_display)

    def test_requires_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = make_scorer(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "init"):
                scorer.get_baseline_score(cast(Context, FakeCtx(FakeModel())))


class MiscTests(unittest.TestCase):
    def test_user_text_format(self) -> None:
        text = CalibratedPIQA._piqa_user_text("goal", "sol-a", "sol-b")
        for fragment in ["Goal: goal", "A. sol-a", "B. sol-b", "letter A or B"]:
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
