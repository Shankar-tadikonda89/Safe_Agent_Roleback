from __future__ import annotations

import dataclasses
import unittest
from collections import Counter

from experiments import phase9_2_simulator as sim


def _fixture(mechanism: str = "transient_tool_fault", family_id: str = "unit_family") -> sim.FamilyConfig:
    return sim.FamilyConfig(
        family_id=family_id, mechanism=mechanism, fault_prior=0.18,
        sensor_noise_sd=0.2, sensor_sensitivity=0.8,
        sensor_false_positive_rate=0.15, repair_probability=0.7,
        late_repair_sensitivity=0.3, irreversible_failure_sensitivity=0.25,
        disruption_failure_sensitivity=0.2, harm_multiplier=1.0,
        action_mix=(0.3, 0.4, 0.3),
    )


class Phase92SimulatorTests(unittest.TestCase):
    def test_scenario_and_outcomes_are_seed_deterministic(self):
        a = sim.generate_scenario(_fixture(), 17, 7001)
        b = sim.generate_scenario(_fixture(), 17, 7001)
        self.assertEqual(a, b)
        self.assertEqual(sim.potential_outcomes(a), sim.potential_outcomes(b))

    def test_streams_are_domain_separated(self):
        seeds = {sim._seed(91, "family", 7, stream) for stream in sim.STREAM_NAMES}
        self.assertEqual(len(seeds), len(sim.STREAM_NAMES))

    def test_four_features_vary_and_remain_in_range(self):
        rows = [sim.generate_scenario(_fixture(), seed, 8001) for seed in range(1200)]
        for name in sim.FEATURE_ORDER:
            values = [row.features[name] for row in rows]
            self.assertGreater(len(set(values)), 1, name)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(tuple(rows[0].features), sim.FEATURE_ORDER)

    def test_alert_sensor_has_false_positives_and_false_negatives(self):
        rows = [sim.generate_scenario(_fixture(), seed, 8002) for seed in range(5000)]
        false_positive = sum(row.latent_fragility < 0.55 and row.prefix.anomaly_alert for row in rows)
        false_negative = sum(row.latent_fragility >= 0.55 and not row.prefix.anomaly_alert for row in rows)
        self.assertGreater(false_positive, 0)
        self.assertGreater(false_negative, 0)

    def test_feature_extractor_is_independent_of_hidden_state_ids_and_outcomes(self):
        scenario = sim.generate_scenario(_fixture(), 25, 8003)
        original = sim.features_from_observation(scenario.prefix)
        altered = dataclasses.replace(
            scenario, latent_fragility=1.0 - scenario.latent_fragility,
            fault_enabled=not scenario.fault_enabled, stale_evidence=not scenario.stale_evidence,
            conflicting_evidence=not scenario.conflicting_evidence,
            root_seed=scenario.root_seed + 100, seed=scenario.seed + 100,
            scenario_hash="changed-id",
        )
        self.assertNotEqual(sim.potential_outcomes(scenario), sim.potential_outcomes(altered))
        self.assertEqual(original, sim.features_from_observation(altered.prefix))
        self.assertEqual(original, scenario.features)

    def test_observed_features_do_not_change_when_realized_fault_is_flipped(self):
        scenario = sim.generate_scenario(_fixture(), 26, 8004)
        flipped = dataclasses.replace(scenario, fault_enabled=not scenario.fault_enabled)
        self.assertEqual(sim.features_from_observation(scenario.prefix),
                         sim.features_from_observation(flipped.prefix))

    def test_mechanisms_have_distinct_transition_rules(self):
        mechanisms = sim.MECHANISMS
        self.assertEqual(len(set(mechanisms)), 8)
        results = {}
        for mechanism in mechanisms:
            config = _fixture(mechanism, f"unit_{mechanism}")
            results[mechanism] = Counter(
                sim.potential_outcomes(sim.generate_scenario(config, seed, 8400))["outcome_class"]
                for seed in range(600)
            )
        self.assertGreater(len({tuple(sorted(values.items())) for values in results.values()}), 1)
        self.assertTrue(any(results[name] != results["transient_tool_fault"]
                            for name in mechanisms if name != "transient_tool_fault"))

    def test_all_outcome_classes_are_supported_by_fixed_unit_fixture(self):
        observed = {
            sim.potential_outcomes(sim.generate_scenario(_fixture(), seed, 8005))["outcome_class"]
            for seed in range(5000)
        }
        self.assertEqual(observed, set(sim.CLASS_NAMES))

    def test_family_configs_are_disjoint_and_test_includes_unseen_mechanisms(self):
        splits = sim.family_configurations()
        id_sets = [{row.family_id for row in rows} for rows in splits.values()]
        for i, left in enumerate(id_sets):
            for right in id_sets[i + 1:]:
                self.assertFalse(left & right)
        train_mechanisms = {row.mechanism for row in splits["training"]}
        test_mechanisms = {row.mechanism for row in splits["test"]}
        self.assertTrue(test_mechanisms - train_mechanisms)

    def test_manifest_contains_exact_split_keys_without_outcomes(self):
        manifest = sim.split_manifest()
        test = manifest["splits"]["test"]
        self.assertEqual(test["scenario_count"], 30 * 256)
        self.assertEqual(len(test["scenario_keys"]), 30 * 256)
        self.assertNotIn("potential_outcomes", manifest)
        self.assertEqual(manifest["class_order"], list(sim.CLASS_NAMES))

    def test_invalid_observable_prefix_is_rejected(self):
        prefix = sim.ObservablePrefix(0.2, False, 0.1, "unknown", 0, 2, 0)
        with self.assertRaises(ValueError):
            sim.features_from_observation(prefix)


if __name__ == "__main__":
    unittest.main()
