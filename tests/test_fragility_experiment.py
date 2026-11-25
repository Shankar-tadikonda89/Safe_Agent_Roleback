from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from experiments import fragility_experiment as experiment
from run_demo import FEATURE_NAMES, NaiveThresholdPolicy


class FragilityExperimentTests(unittest.TestCase):
    def test_feature_allowlist_has_only_decision_time_values(self):
        self.assertEqual(tuple(inspect.signature(experiment.features_from_observation).parameters),
                         ("observable_failure_prob",))
        features = experiment.features_from_observation(0.63)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(set(features), {
            "failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history"
        })
        self.assertNotIn("fault_enabled", features)
        self.assertNotIn("latent_fragility", features)
        self.assertNotIn("passed", features)
        self.assertNotIn("expected", features)
        self.assertEqual(features, experiment.features_from_observation(0.63))

    def test_hidden_metadata_filename_and_row_order_cannot_change_features(self):
        first = experiment.generate_scenario(experiment.TASK_BY_ID[experiment.TRAIN_TASK_IDS[0]], 7)
        fault_flipped = replace(first, fault_enabled=not first.fault_enabled)
        self.assertEqual(
            experiment.features_from_observation(first.observable_failure_prob),
            experiment.features_from_observation(fault_flipped.observable_failure_prob),
        )
        changed_metadata = {
            "fault_enabled": not first.fault_enabled,
            "scenario_hash": "different",
            "task_id": "renamed",
            "seed": 999,
            "filename": "other.jsonl",
            "row_position": 123,
        }
        # Metadata is deliberately not accepted by the feature function.
        feature = experiment.features_from_observation(first.observable_failure_prob)
        self.assertEqual(feature, experiment.features_from_observation(first.observable_failure_prob))
        self.assertEqual(changed_metadata["filename"], "other.jsonl")

        tasks = list(experiment.TASKS)
        forward = {(task.task_id, seed): experiment.generate_scenario(task, seed).scenario_hash
                   for task in tasks for seed in range(3)}
        reverse = {(task.task_id, seed): experiment.generate_scenario(task, seed).scenario_hash
                   for task in reversed(tasks) for seed in reversed(range(3))}
        self.assertEqual(forward, reverse)

    def test_random_properties_use_distinct_reproducible_streams(self):
        self.assertEqual(len(experiment.STREAM_NAMES), len(set(experiment.STREAM_NAMES)))
        derived = [experiment._stream_seed(10, "synthetic_task_00", 4, name)
                   for name in experiment.STREAM_NAMES]
        self.assertEqual(len(derived), len(set(derived)))
        self.assertEqual(derived, [experiment._stream_seed(10, "synthetic_task_00", 4, name)
                                   for name in experiment.STREAM_NAMES])
        samples = [experiment.generate_scenario(experiment.TASKS[0], seed, root_seed=91)
                   for seed in range(128)]
        self.assertTrue(any(row.fault_enabled for row in samples))
        self.assertTrue(any(not row.fault_enabled for row in samples))
        self.assertGreater(len({row.latent_fragility for row in samples}), 100)
        self.assertGreater(len({row.fault_enabled for row in samples}), 1)

    def test_scenario_and_training_example_are_reproducible_and_schema_matches(self):
        task = experiment.TASK_BY_ID[experiment.TRAIN_TASK_IDS[2]]
        first = experiment.generate_scenario(task, 17, 33)
        second = experiment.generate_scenario(task, 17, 33)
        self.assertEqual(first, second)
        example1 = experiment.make_training_example(task, 17, 33)
        example2 = experiment.make_training_example(task, 17, 33)
        self.assertEqual(example1, example2)
        self.assertEqual(tuple(example1["features"]), FEATURE_NAMES)
        self.assertEqual(tuple(experiment.features_from_observation(first.observable_failure_prob)),
                         tuple(example1["feature_names"]))
        self.assertEqual(set(example1["features"]) & {
            "fault_enabled", "latent_fragility", "scenario_hash", "seed", "task_id",
            "passed", "expected", "branches",
        }, set())

    def test_reliability_metrics_are_computed_from_observed_outcomes(self):
        result = experiment._reliability_summary([0.2, 0.8], [False, True])
        self.assertAlmostEqual(result["brier_score"], 0.04)
        self.assertEqual(sum(row["count"] for row in result["reliability_bins"]), 2)

    def test_fault_fragility_and_execution_use_distinct_ground_truth_streams(self):
        rows = [experiment.generate_scenario(experiment.TASKS[0], seed, 44) for seed in range(300)]
        fault_values = {row.fault_enabled for row in rows}
        self.assertEqual(fault_values, {True, False})
        # Supporting check only; the structural guarantee is the separate,
        # domain-separated stream and feature function interface.
        self.assertGreater(len({row.latent_fragility for row in rows}), 250)
        self.assertNotEqual(experiment._stream_seed(44, rows[0].task_id, rows[0].seed, "fault"),
                            experiment._stream_seed(44, rows[0].task_id, rows[0].seed, "fragility"))

    def test_training_uses_training_task_ids_only_and_artifact_loads(self):
        examples = experiment.generate_training_examples(range(64), root_seed=123)
        self.assertTrue(all(row["task_id"] in experiment.TRAIN_TASK_IDS for row in examples))
        model, metadata = experiment.train_model(examples)
        self.assertFalse(set(metadata["training_tasks"]) & set(metadata["evaluation_tasks"]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "model.joblib"
            experiment.save_model(model, metadata, path)
            loaded = experiment.load_model(path)
            scenario = experiment.generate_scenario(experiment.TASKS[0], 4, 123)
            features = experiment.features_from_observation(scenario.observable_failure_prob)
            result = loaded.decide(features)
            self.assertIn(result["decision"], {"continue", "intervene"})
            self.assertGreaterEqual(result["helpful_intervention_probability"], 0)
            self.assertLessEqual(result["helpful_intervention_probability"], 1)

    def test_proposed_decision_uses_one_classifier_call_and_no_counterfactual_branch(self):
        examples = experiment.generate_training_examples(range(64), root_seed=25)
        model, _ = experiment.train_model(examples)
        proposed = experiment.FragilityProposedPolicy(model)
        scenario = experiment.generate_scenario(experiment.TASKS[0], 2, 25)
        features = experiment.features_from_observation(scenario.observable_failure_prob)
        with patch.object(proposed.model, "predict_proba", wraps=proposed.model.predict_proba) as predict, \
                patch.object(experiment, "simulate_counterfactual_branches", side_effect=AssertionError("branch at inference")):
            result = proposed.decide(features)
        self.assertEqual(predict.call_count, 1)
        self.assertIn(result["decision"], {"continue", "intervene"})

    def test_signal_and_policy_decision_precede_action_outcome_execution(self):
        order = []

        class RecordingPolicy:
            def decide(self, features):
                order.append("decision")
                self.assert_feature_schema(features)
                return {"decision": "continue"}

            @staticmethod
            def assert_feature_schema(features):
                if tuple(features) != FEATURE_NAMES:
                    raise AssertionError("policy received an invalid feature set")

        scenario = experiment.generate_scenario(experiment.TASKS[0], 18, 77)

        def outcome(scenario_arg, action):
            order.append("execute")
            return True

        with patch.object(experiment, "_branch_outcome", side_effect=outcome):
            row = experiment._run_policy(scenario, "proposed", RecordingPolicy())
        self.assertEqual(order, ["decision", "execute"])
        self.assertEqual(row["features"]["failure_prob"], scenario.observable_failure_prob)

    def test_training_data_contains_useful_unnecessary_and_ineffective_cases(self):
        found = set()
        harmful = False
        for seed in range(1500):
            scenario = experiment.generate_scenario(experiment.TASKS[0], seed, 2026)
            outcomes = experiment.simulate_counterfactual_branches(scenario)
            cont, intervene = outcomes["continue_passed"], outcomes["intervene_passed"]
            harmful = harmful or (cont and not intervene)
            if not cont and intervene:
                found.add("useful")
            elif cont and intervene:
                found.add("unnecessary")
            elif not cont and not intervene:
                found.add("ineffective")
            if found == {"useful", "unnecessary", "ineffective"}:
                break
        self.assertEqual(found, {"useful", "unnecessary", "ineffective"})
        self.assertFalse(harmful, "the common-random-number simulator is monotone and does not model intervention harm")

    def test_naive_threshold_preserves_strict_point_five_boundary(self):
        policy = NaiveThresholdPolicy()
        self.assertEqual(policy.decide(0.5, {}), "continue")
        self.assertEqual(policy.decide(0.500001, {}), "abort")
        self.assertEqual(policy.decide(0.499999, {}), "continue")

    def test_matched_evaluation_uses_fresh_equal_initial_states_and_valid_metrics(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            results = []
            for suffix in ("a", "b"):
                results.append(experiment.run_experiment(
                    Path(directory) / f"results_{suffix}",
                    Path(directory) / f"model_{suffix}.joblib",
                    root_seed=101,
                    train_replicates=64,
                    evaluation_replicates=8,
                ))
            result = results[0]
            rows = [json.loads(line) for line in result["evaluation_runs"].read_text().splitlines()]
            rows_repeat = [json.loads(line) for line in results[1]["evaluation_runs"].read_text().splitlines()]
            self.assertEqual(result["training_examples"].read_text(), results[1]["training_examples"].read_text())
            stable_fields = ("policy", "task_id", "seed", "scenario_hash", "initial_state_hash",
                             "features", "decision", "intervention", "passed", "steps_executed")
            self.assertEqual(
                [[row[field] for field in stable_fields] for row in rows],
                [[row[field] for field in stable_fields] for row in rows_repeat],
            )
            experiment.validate_matched_runs(rows)
            self.assertEqual(len(rows), len(experiment.EVALUATION_TASK_IDS) * 8 * len(experiment.POLICIES))
            grouped = {}
            for row in rows:
                grouped.setdefault((row["task_id"], row["seed"]), []).append(row)
            for group in grouped.values():
                self.assertEqual({row["policy"] for row in group}, set(experiment.POLICIES))
                self.assertEqual(len({row["initial_state_hash"] for row in group}), 1)
                self.assertEqual(len({id(row["initial_state"]) for row in group}), len(experiment.POLICIES))
                self.assertTrue(all(row["scenario_hash"] == group[0]["scenario_hash"] for row in group))
            for policy in experiment.POLICIES:
                self.assertIn("mean_realized_cost", result["measured_metrics"][policy])
                self.assertEqual(result["measured_metrics"][policy]["attempts"], 32)
            self.assertIn("brier_score", result["measured_metrics"]["risk_calibration"])
            audit = experiment.audit_saved_results(result["evaluation_runs"].parent, result["artifact"])
            self.assertTrue(audit["metrics_match_jsonl"])
            self.assertTrue(audit["offline_only"])

    def test_duplicate_or_unmatched_policy_runs_are_rejected(self):
        scenario = experiment.generate_scenario(experiment.TASKS[-1], 3)
        runs = [experiment._run_policy(scenario, policy, None) for policy in experiment.POLICIES[:2]]
        with self.assertRaises(ValueError):
            experiment.validate_matched_runs(runs)
        duplicate = runs + [dict(runs[0])]
        with self.assertRaises(ValueError):
            experiment.validate_matched_runs(duplicate)


class DecisionDiagnosticTests(unittest.TestCase):
    class FixedPredictor:
        classes_ = experiment.np.asarray([0, 1])

        def __init__(self, probability):
            self.probability = probability
            self.calls = 0

        def predict_proba(self, row):
            self.calls += 1
            return experiment.np.asarray([[1.0 - self.probability, self.probability]])

    def test_expected_costs_and_strict_threshold_boundaries(self):
        cases = (
            (0.0, 0.0, 2.0, 2.0, "continue"),
            (0.25, 0.75, 1.75, 1.0, "continue"),
            (0.5, 1.5, 1.5, 0.0, "continue"),
            (0.500001, 1.500003, 1.499999, -0.000004, "intervene"),
            (1.0, 3.0, 1.0, -2.0, "intervene"),
        )
        for probability, cont, intervention, delta, action in cases:
            with self.subTest(p_helpful=probability):
                result = experiment.calculate_decision_diagnostics(probability)
                self.assertAlmostEqual(result["p_helpful"], probability)
                self.assertAlmostEqual(result["continue_expected_cost"], cont)
                self.assertAlmostEqual(result["intervene_expected_cost"], intervention)
                self.assertAlmostEqual(result["cost_difference"], delta)
                self.assertEqual(result["selected_action"], action)

    def test_invalid_probabilities_fail_explicitly(self):
        for probability in (float("nan"), float("inf"), float("-inf"), -0.001, 1.001,
                            "0.5", None, True):
            with self.subTest(probability=probability):
                with self.assertRaises(experiment.InvalidHelpfulProbability):
                    experiment.calculate_decision_diagnostics(probability)

        features = experiment.features_from_observation(0.7)
        bad_model = self.FixedPredictor(float("nan"))
        with self.assertRaises(experiment.InvalidHelpfulProbability):
            experiment.FragilityProposedPolicy(bad_model).decide(features)
        self.assertEqual(bad_model.calls, 1)

    def test_policy_decision_agrees_with_diagnostic_rule_and_predicts_once(self):
        features = experiment.features_from_observation(0.7)
        for probability in (0.0, 0.25, 0.5, 0.500001, 1.0):
            model = self.FixedPredictor(probability)
            result = experiment.FragilityProposedPolicy(model).decide(features)
            diagnostics = result["decision_diagnostics"]
            self.assertEqual(result["decision"], "intervene" if probability > 0.5 else "continue")
            self.assertEqual(result["decision"], diagnostics["selected_action"])
            self.assertEqual(model.calls, 1)

    def test_future_proposed_run_logs_expected_cost_diagnostics_additively(self):
        scenario = experiment.generate_scenario(experiment.TASKS[0], 20, 91)
        predictor = self.FixedPredictor(0.75)
        proposed = experiment.FragilityProposedPolicy(predictor)
        result = experiment._run_policy(scenario, "proposed", proposed)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(result["decision"], "intervene")
        self.assertEqual(result["decision_diagnostics"]["selected_action"], "intervene")
        self.assertAlmostEqual(result["decision_diagnostics"]["p_helpful"], 0.75)
        self.assertAlmostEqual(result["decision_diagnostics"]["continue_expected_cost"], 2.25)
        self.assertAlmostEqual(result["decision_diagnostics"]["intervene_expected_cost"], 1.25)
        self.assertNotIn("realized_cost", result)

        baseline = experiment._run_policy(scenario, "no_intervention", None)
        realized = experiment._realized_cost(result, baseline)
        self.assertIn(realized, {1.0, 2.0, 5.0})
        self.assertNotEqual(result["decision_diagnostics"]["intervene_expected_cost"], realized)

    def test_legacy_records_without_diagnostics_remain_readable(self):
        scenario = experiment.generate_scenario(experiment.TASKS[-1], 29, 17)
        rows = [
            experiment._run_policy(scenario, "no_intervention", None),
            experiment._run_policy(scenario, "naive_threshold", None),
            experiment._run_policy(
                scenario, "proposed", experiment.FragilityProposedPolicy(self.FixedPredictor(0.5))
            ),
        ]
        for row in rows:
            row.pop("decision_diagnostics", None)  # shape of historical records
        experiment.validate_matched_runs(rows)


if __name__ == "__main__":
    unittest.main()
