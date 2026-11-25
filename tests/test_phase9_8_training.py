import unittest

from experiments import phase9_7_simulator as sim
from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_8_training import (
    C_GRID,
    PARTITIONS,
    VARIANTS,
    _generate_pair,
    _select_c,
    load_and_validate_design,
    Phase98DataError,
)


class Phase98TrainingProtocolTests(unittest.TestCase):
    def test_frozen_hashes_and_disjoint_group_manifest(self):
        configs, manifest, _ = load_and_validate_design()
        self.assertEqual(len(configs["configurations"]), 120)
        self.assertEqual(len(manifest["groups"]), 120)
        owners = {}
        for group in manifest["groups"]:
            prior = owners.setdefault(group["configuration_id"], group["partition"])
            self.assertEqual(prior, group["partition"])
        self.assertEqual(
            {part: sum(g["partition"] == part for g in manifest["groups"])
             for part in manifest["partition_counts"]},
            manifest["partition_counts"],
        )

    def test_only_authorized_generation_partitions_are_named(self):
        self.assertEqual(PARTITIONS, ("fit", "calibration", "validation"))
        _, manifest, _ = load_and_validate_design()
        present = {g["partition"] for g in manifest["groups"]}
        self.assertTrue(set(PARTITIONS).issubset(present))

    def test_feature_and_class_order_match_frozen_schema(self):
        self.assertEqual(FEATURE_ORDER, (
            "failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history",
        ))
        self.assertEqual(sim.OUTCOME_CLASSES, (
            "recovery", "unnecessary_intervention", "ineffective_intervention", "intervention_harm",
        ))
        self.assertEqual(len(VARIANTS), 6)
        self.assertEqual(VARIANTS["full_features"], FEATURE_ORDER)

    def test_configured_pair_is_deterministic_and_separate(self):
        configs, manifest, _ = load_and_validate_design()
        group = next(g for g in manifest["groups"] if g["partition"] == "fit")
        config = next(c for c in configs["configurations"]
                      if c["configuration_id"] == group["configuration_id"])
        observation_a, outcome_a = _generate_pair(group, 0, config)
        observation_b, outcome_b = _generate_pair(group, 0, config)
        self.assertEqual(observation_a, observation_b)
        self.assertEqual(outcome_a, outcome_b)
        self.assertEqual(observation_a["feature_names"], list(FEATURE_ORDER))
        self.assertEqual(tuple(observation_a["features"]), FEATURE_ORDER)
        self.assertNotIn("outcome_class", observation_a)
        self.assertNotIn("features", outcome_a)
        self.assertTrue(outcome_a["evaluator_only"])

    def test_non_authorized_partition_is_rejected_before_scenario_generation(self):
        configs, manifest, _ = load_and_validate_design()
        group = next(g for g in manifest["groups"] if g["partition"] == "heldout")
        config = next(c for c in configs["configurations"]
                      if c["configuration_id"] == group["configuration_id"])
        with self.assertRaises(Phase98DataError):
            _generate_pair(group, 0, config)

    def test_c_selection_uses_only_log_loss_and_breaks_ties_to_lower_c(self):
        results = [
            {"C": 10.0, "validation_multiclass_log_loss": 0.8},
            {"C": 1.0, "validation_multiclass_log_loss": 0.7},
            {"C": 0.1, "validation_multiclass_log_loss": 0.7},
        ]
        self.assertEqual(_select_c(results)["C"], 0.1)
        self.assertEqual(set(C_GRID), {result["C"] for result in results})
        with self.assertRaises(ValueError):
            _select_c([{"C": 1.0, "validation_multiclass_log_loss": float("nan")}])


if __name__ == "__main__":
    unittest.main()
