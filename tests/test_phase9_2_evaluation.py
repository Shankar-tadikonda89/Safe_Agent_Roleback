from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments import phase9_2_evaluation as ev
from experiments import phase9_2_simulator as sim


class FakeEstimator:
    classes_ = np.asarray([0, 1, 2, 3])
    def __init__(self, result):
        self.result = result
        self.calls = 0
    def predict_proba(self, _x):
        self.calls += 1
        return self.result


def _fixture():
    return sim.FamilyConfig(
        family_id="eval_unit", mechanism="transient_tool_fault", fault_prior=0.18,
        sensor_noise_sd=0.2, sensor_sensitivity=0.8, sensor_false_positive_rate=0.15,
        repair_probability=0.7, late_repair_sensitivity=0.3,
        irreversible_failure_sensitivity=0.25, disruption_failure_sensitivity=0.2,
        harm_multiplier=1.0, action_mix=(0.3, 0.4, 0.3),
    )


class Phase92EvaluationTests(unittest.TestCase):
    def test_expected_cost_rule_and_tie(self):
        tie = ev.calculate_expected_costs([2 / 3, 1 / 3, 0, 0])
        self.assertAlmostEqual(tie["expected_continue_cost"], 2.0)
        self.assertAlmostEqual(tie["expected_intervene_cost"], 2.0)
        self.assertEqual(tie["selected_action"], "continue")
        recover = ev.calculate_expected_costs([0.8, 0.1, 0.1, 0.0])
        self.assertEqual(recover["selected_action"], "intervene")

    def test_public_decide_rejects_malformed_raw_probabilities(self):
        features = sim.generate_scenario(_fixture(), 5, 8010).features
        for invalid in (
            [["0.25", "0.25", "0.25", "0.25"]],
            [[True, False, False, False]],
            [[float("nan"), 0.2, 0.3, 0.5]],
            [[-0.1, 0.4, 0.4, 0.3]],
            [[0.1, 0.2, 0.3, 0.3]],
            [[0.2, 0.8]],
        ):
            policy = ev.ProposedPolicy(FakeEstimator(invalid))
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                policy.decide(dict(features))

    def test_valid_public_prediction_calls_estimator_once(self):
        estimator = FakeEstimator([[0.7, 0.1, 0.15, 0.05]])
        result = ev.ProposedPolicy(estimator).decide(sim.generate_scenario(_fixture(), 2, 8011).features)
        self.assertEqual(estimator.calls, 1)
        self.assertIn(result["selected_action"], {"continue", "intervene"})
        self.assertEqual(result["class_names"], list(sim.CLASS_NAMES))

    def test_frozen_design_has_disjoint_family_splits_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            prereg, manifest = root / "prereg.md", root / "split.json"
            result = ev.prepare_design(prereg, manifest)
            before = hashlib.sha256(manifest.read_bytes()).hexdigest()
            digest, loaded = ev._load_frozen_design(prereg, manifest)
            self.assertEqual(digest, result["manifest_sha256"])
            split = loaded["splits"]
            ids = [{r["family_id"] for r in split[name]["family_configurations"]}
                   for name in ("training", "validation", "pilot", "test")]
            for i, left in enumerate(ids):
                for right in ids[i + 1:]:
                    self.assertFalse(left & right)
            with self.assertRaises(FileExistsError):
                ev.prepare_design(prereg, manifest)
            self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), before)

    def test_saved_artifact_checksum_pipeline_and_class_order(self):
        x = np.asarray([
            [0.1, 0.2, 0.1, 0.0], [0.2, 0.5, 0.3, 0.3],
            [0.4, 0.8, 0.7, 0.6], [0.9, 0.2, 0.5, 1.0],
            [0.3, 0.65, 0.4, 0.0], [0.7, 0.95, 0.9, 0.7],
            [0.8, 0.2, 0.25, 0.3], [0.55, 0.65, 0.8, 0.9],
        ])
        y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(random_state=0))]).fit(x, y)
        metadata = {"experiment_version": sim.EXPERIMENT_VERSION,
                    "feature_names": list(ev.VARIANTS["full"]),
                    "class_names": list(sim.CLASS_NAMES), "variant": "full"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path, checksum = ev._save_artifact(model, metadata, Path(temp), "full")
            loaded, loaded_meta, loaded_sha = ev._load_artifact(path)
            self.assertEqual(checksum, loaded_sha)
            self.assertEqual(loaded_meta, metadata)
            self.assertEqual(loaded.named_steps["classifier"].classes_.tolist(), [0, 1, 2, 3])
            self.assertEqual(loaded.named_steps["scale"].n_features_in_, 4)

    def test_matched_policy_runs_outcomes_and_prefix_overhead(self):
        scenarios = [sim.generate_scenario(_fixture(), seed, 8012) for seed in (1, 2, 3)]
        model = ev.ProposedPolicy(FakeEstimator([[0.25, 0.25, 0.25, 0.25]]))
        rows, potentials = ev._run_scenarios(scenarios, model, "continue", 0.1, "unit-hash")
        ev._validate_rows(rows, potentials)
        self.assertEqual(len(rows), len(scenarios) * len(sim.POLICY_NAMES))
        prefix_rows = [r for r in rows if r["policy"] == "prefix_branching_approximation"]
        self.assertTrue(all(r["branch_rollouts"] == 8 for r in (x["branch_diagnostics"] for x in prefix_rows)))
        self.assertTrue(all(r["branch_overhead_steps"] == 8 for r in prefix_rows))
        for row in rows:
            self.assertEqual(row["success"], row["selected_outcome_passed"])

    def test_bootstrap_is_reproducible(self):
        values = {f"family_{i}": [float(i), float(i + 1)] for i in range(8)}
        self.assertEqual(ev._cluster_ci(values), ev._cluster_ci(values))

    def test_evaluation_function_does_not_call_fit_or_training(self):
        source = __import__("inspect").getsource(ev.run_evaluation)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("train_and_save(", source)
        self.assertIn("_load_verified_models", source)


if __name__ == "__main__":
    unittest.main()
