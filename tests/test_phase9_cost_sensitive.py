from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import phase9_cost_sensitive as phase9


def _scenario_for_class(family: str, outcome_class: str, start: int = 0,
                        root_seed: int = 24680, harm_probability: float = 0.5):
    for seed in range(start, start + 20_000):
        scenario = phase9.generate_scenario(family, seed, root_seed, harm_probability)
        if phase9.potential_outcomes(scenario)["outcome_class"] == outcome_class:
            return scenario
    raise AssertionError(f"could not find deterministic fixture for {outcome_class}")


class FixedFourClassPredictor:
    classes_ = np.asarray([0, 1, 2, 3])

    def __init__(self, probabilities=(0.25, 0.25, 0.25, 0.25)):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.calls = 0

    def predict_proba(self, features):
        self.calls += 1
        return self.probabilities.reshape(1, -1)


class PhaseNineTests(unittest.TestCase):
    def test_scenario_harm_and_potential_outcomes_are_deterministic(self):
        first = phase9.generate_scenario("family_a", 17, 100, 0.1)
        second = phase9.generate_scenario("family_a", 17, 100, 0.1)
        self.assertEqual(first, second)
        self.assertEqual(phase9.potential_outcomes(first), phase9.potential_outcomes(second))
        self.assertEqual(
            first.harm_uniform,
            random.Random(phase9._stream_seed(100, "family_a", 17, "intervention_harm")).random(),
        )

    def test_random_streams_are_domain_separated(self):
        seeds = {
            phase9._stream_seed(99, "family", 5, name)
            for name in phase9.STREAM_NAMES
        }
        self.assertEqual(len(seeds), len(phase9.STREAM_NAMES))
        self.assertNotEqual(
            phase9._stream_seed(99, "family", 5, "intervention_harm"),
            phase9._stream_seed(99, "family", 5, "execution"),
        )

    def test_all_four_potential_outcome_categories_are_possible(self):
        observed = {
            phase9.potential_outcomes(
                phase9.generate_scenario("category_fixture", seed, 123, 0.5)
            )["outcome_class"]
            for seed in range(20_000)
        }
        self.assertEqual(observed, set(phase9.CLASS_NAMES))

    def test_generation_rejects_duplicate_or_overlapping_families(self):
        with self.assertRaises(ValueError):
            phase9.generate_training_examples(families=("train_x", "train_x"))
        with self.assertRaises(ValueError):
            phase9.generate_training_examples(families=(phase9.EVALUATION_FAMILIES[0],))
        with self.assertRaises(ValueError):
            phase9.generate_evaluation_scenarios(
                0.1, families=(phase9.TRAIN_FAMILIES[0],), root_seeds=(1,), replicates_per_family=1
            )

    def test_training_labels_use_outcomes_but_features_are_allowlisted(self):
        examples = phase9.generate_training_examples(replicates_per_family=8)
        self.assertEqual(len(examples), len(phase9.TRAIN_FAMILIES) * 8)
        self.assertTrue(all(row["feature_names"] == list(phase9.FEATURE_NAMES) for row in examples))
        for row in examples:
            self.assertEqual(set(row["features"]), set(phase9.FEATURE_NAMES))
            outcomes = row["potential_outcomes"]
            expected = phase9.CLASS_NAMES[
                phase9._category_index(outcomes["continue_passed"], outcomes["intervene_passed"])
            ]
            self.assertEqual(row["outcome_class"], expected)
            self.assertNotIn("potential_outcomes", row["features"])
            self.assertNotIn("fault_enabled", row["features"])

    def test_missing_training_class_fails_and_class_order_is_validated(self):
        examples = [
            {
                "scenario_key": {"task_family": "train_fixture"},
                "feature_names": list(phase9.FEATURE_NAMES),
                "features": dict(zip(phase9.FEATURE_NAMES, (0.4 + 0.1 * i, 0.25, 0.0, 0.0))),
                "outcome_class_index": i,
            }
            for i in range(4)
        ]
        model, _ = phase9.train_model(examples, training_families=("train_fixture",))
        self.assertTrue(np.array_equal(model.classes_, np.arange(4)))
        with self.assertRaisesRegex(ValueError, "missing required outcome classes"):
            phase9.train_model(examples[:-1], training_families=("train_fixture",))
        bad = FixedFourClassPredictor()
        bad.classes_ = np.asarray([0, 1, 3, 2])
        with self.assertRaisesRegex(ValueError, "class indices"):
            phase9.ProposedPolicy(bad)

    def test_probability_validation_shape_bounds_finiteness_and_sum(self):
        for invalid in (
            [0.2, 0.3, 0.5],
            [[0.25, 0.25, 0.25, 0.25]],
            [float("nan"), 0.2, 0.3, 0.5],
            [float("inf"), 0.0, 0.0, 0.0],
            [-0.01, 0.5, 0.5, 0.01],
            [1.01, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.3],
            ["0.25", 0.25, 0.25, 0.25],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(phase9.InvalidOutcomeProbabilities):
                    phase9.calculate_expected_costs(invalid)
        accepted = phase9._validate_probability_vector([0.25, 0.25, 0.25, 0.25 + 1e-9])
        self.assertAlmostEqual(float(accepted.sum()), 1.0 + 1e-9)
        wrong_shape_model = FixedFourClassPredictor()
        wrong_shape_model.predict_proba = lambda _: np.asarray([[0.2, 0.8]])
        with self.assertRaises(phase9.InvalidOutcomeProbabilities):
            phase9.ProposedPolicy(wrong_shape_model).decide(phase9.feature_vector(0.4))

    def test_expected_cost_formulas_and_continue_on_equality(self):
        tie = phase9.calculate_expected_costs([2 / 3, 1 / 3, 0.0, 0.0])
        self.assertAlmostEqual(tie["expected_continue_cost"], 2.0)
        self.assertAlmostEqual(tie["expected_intervene_cost"], 2.0)
        self.assertEqual(tie["selected_action"], "continue")
        rng = random.Random(501)
        for _ in range(500):
            raw = [rng.random() for _ in phase9.CLASS_NAMES]
            total = sum(raw)
            q = [value / total for value in raw]
            result = phase9.calculate_expected_costs(q)
            q_r, _q_u, q_n, q_h = q
            self.assertAlmostEqual(result["expected_continue_cost"], 3 * (q_r + q_n))
            self.assertAlmostEqual(result["expected_intervene_cost"], 2 + 3 * (q_n + q_h))
            by_cost = result["expected_intervene_cost"] < result["expected_continue_cost"]
            equivalent = 3 * q_r > 2 + 3 * q_h
            self.assertEqual(by_cost, equivalent)

    def test_cost_table_and_overhead_are_applied_once(self):
        for name in phase9.CLASS_NAMES:
            self.assertEqual(phase9.CONTINUE_COSTS[name], {"recovery": 3, "unnecessary_intervention": 0,
                                                           "ineffective_intervention": 3, "intervention_harm": 0}[name])
            self.assertEqual(phase9.INTERVENE_COSTS[name], {"recovery": 2, "unnecessary_intervention": 2,
                                                            "ineffective_intervention": 5, "intervention_harm": 5}[name])
        for category in phase9.CLASS_NAMES:
            scenario = _scenario_for_class("cost_fixture", category)
            outcomes = phase9.potential_outcomes(scenario)
            self.assertEqual(outcomes["continue_realized_cost"], phase9.CONTINUE_COSTS[category])
            self.assertEqual(outcomes["intervene_realized_cost"], phase9.INTERVENE_COSTS[category])
            self.assertEqual(outcomes["cost_components"]["intervention_overhead_cost"], 2.0)
            self.assertTrue(outcomes["cost_components"]["overhead_included_in_intervention_cost"])
            self.assertEqual(outcomes["cost_components"]["additional_harm_surcharge"], 0.0)

    def test_inference_receives_only_features_and_never_branches(self):
        model = FixedFourClassPredictor([0.4, 0.2, 0.2, 0.2])
        policy = phase9.ProposedPolicy(model)
        features = phase9.feature_vector(0.6)
        with patch.object(phase9, "potential_outcomes", side_effect=AssertionError("inference branched")):
            result = policy.decide(features)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["class_names"], list(phase9.CLASS_NAMES))
        with self.assertRaises(ValueError):
            policy.decide({**features, "outcome_class": "recovery"})

    def test_matched_runs_are_isolated_and_selected_outcomes_match(self):
        scenarios = [
            _scenario_for_class(family, category)
            for family in ("eval_fixture_a", "eval_fixture_b")
            for category in phase9.CLASS_NAMES
        ]
        model = FixedFourClassPredictor([0.9, 0.05, 0.05, 0.0])
        rows, potential = phase9._run_policies_after_decisions(
            scenarios, phase9.ProposedPolicy(model), "fixture-checksum"
        )
        phase9._validate_record_sets(rows, potential)
        grouped = {}
        for row in rows:
            grouped.setdefault(tuple(row["scenario_key"].values()), []).append(row)
        self.assertEqual(len(rows), len(scenarios) * 3)
        for matched in grouped.values():
            self.assertEqual({row["policy"] for row in matched}, set(phase9.POLICY_NAMES))
            self.assertEqual(len({id(row["initial_state"]) for row in matched}), 3)
            self.assertEqual(len({row["initial_state_hash"] for row in matched}), 1)
            for row in matched:
                self.assertEqual(row["selected_outcome_passed"], row["success"])
        with self.assertRaises(ValueError):
            phase9._validate_record_sets(rows + [dict(rows[0])], potential)

    def test_metrics_class_order_empty_bins_clipping_and_undefined_scores(self):
        scenarios = [
            _scenario_for_class(family, category)
            for family in ("metric_family_a", "metric_family_b")
            for category in phase9.CLASS_NAMES
        ]
        model = FixedFourClassPredictor([1.0, 0.0, 0.0, 0.0])
        rows, potential = phase9._run_policies_after_decisions(
            scenarios, phase9.ProposedPolicy(model), "metric-checksum"
        )
        metrics = phase9.calculate_metrics(rows, potential, bootstrap_resamples=100, bootstrap_seed=1)
        quality = metrics["probability_quality"]
        self.assertEqual(quality["confusion_matrix_actual_rows_predicted_columns"][0][0], 2)
        self.assertEqual(quality["confusion_matrix_actual_rows_predicted_columns"][3][0], 2)
        self.assertTrue(math.isfinite(quality["multiclass_log_loss"]))
        self.assertEqual(quality["log_loss_clip_epsilon"], 1e-15)
        for name in phase9.CLASS_NAMES:
            bins = quality["per_class"][name]["reliability_bins"]
            self.assertEqual(len(bins), 10)
            self.assertTrue(any(row["count"] == 0 and row["mean_predicted"] is None for row in bins))
        self.assertIsNone(quality["per_class"]["intervention_harm"]["precision"])

    def test_probability_scores_keep_argmax_and_policy_action_distinct(self):
        probs = [0.10, 0.80, 0.05, 0.05]
        diag = phase9.calculate_expected_costs(probs)
        self.assertEqual(diag["selected_action"], "continue")
        self.assertEqual(int(np.argmax(probs)), 1)

    def test_audit_reproduces_from_jsonl_without_deserializing_model(self):
        training = phase9.generate_training_examples()
        model, metadata = phase9.train_model(training)
        scenarios = [
            _scenario_for_class(family, category, root_seed=20261101, harm_probability=0.1)
            for family in (phase9.EVALUATION_FAMILIES[0], phase9.EVALUATION_FAMILIES[1])
            for category in phase9.CLASS_NAMES
        ]
        rows, potential = phase9._run_policies_after_decisions(
            scenarios, phase9.ProposedPolicy(model), "temporary"
        )
        for row in rows:
            row["local_latency_seconds"] = 0.0
        metrics = phase9.calculate_metrics(rows, potential)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            artifact = phase9.save_model(model, metadata, root / "model.joblib")
            checksum = hashlib.sha256(artifact["artifact"].read_bytes()).hexdigest()
            for row in rows:
                if row["policy"] == "proposed":
                    row["model_sha256"] = checksum
            prereg = root / "preregistration.md"
            prereg.write_text("frozen test preregistration\n", encoding="utf-8")
            output_a, output_b = root / "run_a", root / "run_b"
            phase9._write_outputs(output_a, rows, potential, metrics, 0.1, checksum, prereg, training)
            phase9._write_outputs(output_b, rows, potential, metrics, 0.1, checksum, prereg, training)
            with patch.object(phase9.joblib, "load", side_effect=AssertionError("audit deserialized model")):
                audit_a = phase9.audit_results(output_a, artifact["artifact"])
                audit_b = phase9.audit_results(output_b, artifact["artifact"])
            self.assertEqual(audit_a, audit_b)
            self.assertTrue(audit_a["metrics_match_jsonl"])
            self.assertFalse(audit_a["model_deserialized"])

    def test_existing_output_and_model_paths_are_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            output = root / "occupied"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                phase9._ensure_fresh_output(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            artifact = root / "already.joblib"
            artifact.write_bytes(b"prior")
            with self.assertRaises(FileExistsError):
                phase9._ensure_fresh_output(root / "new_output", artifact)
            self.assertEqual(artifact.read_bytes(), b"prior")


if __name__ == "__main__":
    unittest.main()
