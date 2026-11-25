from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.phase9_5_observations import FEATURE_ORDER, extract_features
from experiments.phase9_7_simulator import PolicyScenario
from experiments.phase9_8_decision_freeze import (
    DecisionFreezeError,
    FROZEN_DECISION_SCHEMA_VERSION,
    FROZEN_RAW_HASHES,
    FrozenPilotDesign,
    INITIAL_STATE_HASH,
    _scenario_id,
    _persist_decision_freeze,
    _verify_decision_freeze,
    load_frozen_pilot_design,
)
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES,
    PilotDecision,
    PilotScenario,
)

from tests.test_phase9_8_pilot_runner import _policy_scenario


class DecisionFreezeTests(unittest.TestCase):
    def test_frozen_pilot_membership_loads_without_building_scenarios(self):
        design = load_frozen_pilot_design(Path.cwd())
        self.assertEqual(len(design.scenario_specs), 2560)
        self.assertEqual(len({spec[0] for spec in design.scenario_specs.values()}), 10)

    def setUp(self):
        self.scenarios = []
        self.specs = {}
        self.configuration_mechanisms = {"transient_tool_failure-cfg-20": "transient_tool_failure"}
        table = __import__("json").loads(Path("results/phase9_8_configuration_table.json").read_text(encoding="utf-8"))
        self.configuration_parameters = {
            "transient_tool_failure-cfg-20": next(
                row["parameters"] for row in table["configurations"]
                if row["configuration_id"] == "transient_tool_failure-cfg-20"
            ),
        }
        self.artifact_hashes = dict(FROZEN_RAW_HASHES)
        for seed in (0, 1):
            scenario_id = _scenario_id(20269804, seed, "transient_tool_failure")
            base = _policy_scenario(scenario_id)
            policy = PolicyScenario(scenario_id, INITIAL_STATE_HASH, base.snapshot)
            scenario = PilotScenario(policy, "transient_tool_failure-cfg-20",
                                     "transient_tool_failure", 20269804, seed)
            self.scenarios.append(scenario)
            self.specs[scenario_id] = (
                "transient_tool_failure-cfg-20", "transient_tool_failure", 20269804, seed,
            )
        self.design = FrozenPilotDesign(self.specs, self.configuration_mechanisms,
                                        self.configuration_parameters,
                                        self.artifact_hashes)
        self.decisions = self._make_decisions()

    def _make_decisions(self):
        decisions = []
        for scenario in self.scenarios:
            features = extract_features(scenario.policy_scenario.snapshot.observation).as_ordered_dict()
            for policy in COMPARATIVE_POLICIES:
                decisions.append(PilotDecision(
                    scenario_id=scenario.policy_scenario.scenario_id,
                    configuration_group_id=scenario.configuration_group_id,
                    mechanism=scenario.mechanism, root_seed=scenario.root_seed,
                    scenario_seed=scenario.scenario_seed,
                    initial_state_hash=scenario.policy_scenario.initial_state_hash,
                    policy=policy, selected_action="continue",
                    decision_sequence=scenario.policy_scenario.snapshot.observation.provenance.decision_sequence,
                    features=tuple((name, float(features[name])) for name in FEATURE_ORDER),
                    diagnostics=(),
                ))
        return decisions

    def test_persist_and_reread_validate_complete_versioned_freeze(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            directory, digest = _persist_decision_freeze(
                self.decisions, self.scenarios, design=self.design, output_root=tmp,
            )
            verified = _verify_decision_freeze(
                directory, self.scenarios, design=self.design,
                expected_manifest_sha256=digest,
            )
            self.assertEqual(len(verified.decisions), 2 * len(COMPARATIVE_POLICIES))
            self.assertEqual(verified.manifest_sha256, digest)
            self.assertIn(FROZEN_DECISION_SCHEMA_VERSION,
                          (directory / "decisions.jsonl").read_text(encoding="utf-8"))
            with self.assertRaises(DecisionFreezeError):
                _persist_decision_freeze(
                    self.decisions, self.scenarios, design=self.design, output_root=tmp,
                )

    def test_incomplete_duplicate_and_mismatched_provenance_rejected_before_write(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            invalid = (
                self.decisions[:-1],
                self.decisions + [self.decisions[0]],
                [PilotDecision(**{**self.decisions[0].__dict__, "initial_state_hash": "bad"})]
                + self.decisions[1:],
            )
            for rows in invalid:
                with self.subTest(count=len(rows)), self.assertRaises(DecisionFreezeError):
                    _persist_decision_freeze(rows, self.scenarios,
                                            design=self.design, output_root=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_scenario_group_and_frozen_provenance_membership_is_enforced(self):
        bad = list(self.scenarios)
        original = bad[0]
        bad[0] = PilotScenario(original.policy_scenario, "unknown-group",
                               original.mechanism, original.root_seed, original.scenario_seed)
        with self.assertRaises(DecisionFreezeError):
            self.design.validate_scenarios(bad)

    def test_tampered_or_missing_records_and_wrong_manifest_anchor_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            directory, digest = _persist_decision_freeze(
                self.decisions, self.scenarios, design=self.design, output_root=tmp,
            )
            record_path = directory / "decisions.jsonl"
            original = record_path.read_bytes()
            record_path.write_bytes(original + b"{}\n")
            with self.assertRaises(DecisionFreezeError):
                _verify_decision_freeze(directory, self.scenarios, design=self.design,
                                       expected_manifest_sha256=digest)
            record_path.write_bytes(original)
            with self.assertRaises(DecisionFreezeError):
                _verify_decision_freeze(directory, self.scenarios, design=self.design,
                                       expected_manifest_sha256="0" * 64)
            record_path.unlink()
            with self.assertRaises(DecisionFreezeError):
                _verify_decision_freeze(directory, self.scenarios, design=self.design,
                                       expected_manifest_sha256=digest)


if __name__ == "__main__":
    unittest.main()
