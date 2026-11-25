from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from experiments import phase9_2_simulator as phase92
from experiments.phase9_5_event_history import InvalidEventHistory, TrajectoryEvent
from experiments.phase9_5_observations import (
    FEATURE_ORDER,
    OBSERVATION_SCHEMA_VERSION,
    RISK_SIGNAL_KIND,
    InvalidObservation,
    ObservableRiskEvidence,
    extract_features,
)
from experiments.phase9_7_runner import (
    PHASE97_ARTIFACT_VERSION,
    PHASE97_COST_MATRIX,
    PHASE97_FEATURE_SCHEMA,
    PHASE97_RISK_SIGNAL_ID,
    POLICY_NAMES,
    Phase97ProposedPolicyAdapter,
    decide,
    expected_costs,
    record_applied_intervention,
    validate_phase97_artifact_metadata,
)
from experiments.phase9_7_simulator import (
    CLASS_TO_INDEX,
    CONTINUE_COSTS,
    FAILURE_HORIZON,
    FAILURE_PENALTY,
    INTERVENTION_OVERHEAD_COST,
    INTERVENE_COSTS,
    MECHANISMS,
    OUTCOME_CLASSES,
    PolicyScenario,
    RISK_SOURCE_ID,
    RISK_SOURCE_VERSION,
    DecisionSnapshot,
    EvaluatorCase,
    InvalidScenario,
    TimedEvent,
    TimedTrajectoryLedger,
    _rng,
    _stream_seed,
    append_applied_intervention,
    build_scenario,
    capture_decision_snapshot,
    classify_outcome,
    evaluate_potential_outcomes,
    reserve_new_output_path,
)


def _evidence(
    *, anomaly: float = 0.4, alert: bool = False, errors: int = 0,
    warnings: int = 0, observed: int = -1,
    source: str = RISK_SOURCE_ID, version: str = RISK_SOURCE_VERSION,
) -> ObservableRiskEvidence:
    return ObservableRiskEvidence(
        anomaly_score=anomaly,
        anomaly_alert=alert,
        visible_error_count=errors,
        visible_warning_count=warnings,
        source_id=source,
        source_version=version,
        observed_through_sequence=observed,
    )


def _snapshot(*, evidence: ObservableRiskEvidence | None = None,
              ledger: TimedTrajectoryLedger | None = None,
              decision_tick: int = 0, current: int = 0, planned: int = 2,
              action: str = "reversible_write", observed: int | None = None,
              decision_sequence: int | None = None) -> DecisionSnapshot:
    ledger = ledger or TimedTrajectoryLedger("trajectory-test")
    d = len(ledger.events) if decision_sequence is None else decision_sequence
    o = d - 1 if observed is None else observed
    return capture_decision_snapshot(
        evidence=evidence or _evidence(observed=o),
        pending_action_type=action,
        plan_version="visible-plan-v1",
        current_step_index=current,
        planned_step_count=planned,
        ledger=ledger,
        evidence_timestamp_tick=decision_tick,
        decision_timestamp_tick=decision_tick,
        observed_through_sequence=o,
        decision_sequence=d,
        observation_id="test-snapshot",
    )


def _outcome_case(scenario_id: str, mechanism: str, outcome: str) -> EvaluatorCase:
    if outcome == "recovery":
        return EvaluatorCase(scenario_id, mechanism, 1.0, 0.0, 0.0, 1.0)
    if outcome == "unnecessary_intervention":
        return EvaluatorCase(scenario_id, mechanism, 0.0, 1.0, 1.0, 1.0)
    if outcome == "ineffective_intervention":
        return EvaluatorCase(scenario_id, mechanism, 1.0, 0.0, 1.0, 1.0)
    if outcome == "intervention_harm":
        return EvaluatorCase(scenario_id, mechanism, 0.0, 1.0, 1.0, 0.0)
    raise AssertionError(outcome)


class _FakeProposedPolicy:
    def __init__(self, action: str = "continue") -> None:
        self.action = action
        self.calls: list[dict[str, float]] = []

    def decide(self, features: dict[str, float]) -> dict[str, str]:
        self.calls.append(dict(features))
        return {"selected_action": self.action}


def _validated_adapter(fake: _FakeProposedPolicy) -> Phase97ProposedPolicyAdapter:
    metadata = {
        "artifact_version": PHASE97_ARTIFACT_VERSION,
        "feature_schema": PHASE97_FEATURE_SCHEMA,
        "risk_signal_id": PHASE97_RISK_SIGNAL_ID,
        "feature_names": list(FEATURE_ORDER),
        "class_names": list(OUTCOME_CLASSES),
        "cost_matrix": PHASE97_COST_MATRIX,
        "failure_target": "next_verifier_checkpoint_rejects_without_intervention",
        "failure_horizon": "next_verifier_checkpoint_or_final_verification",
    }
    return Phase97ProposedPolicyAdapter(fake, metadata)


class Phase97ScenarioAndProvenanceTests(unittest.TestCase):
    def test_seeded_scenario_and_all_public_fields_are_reproducible(self):
        a = build_scenario(root_seed=971, scenario_seed=3, mechanism=MECHANISMS[0])
        b = build_scenario(root_seed=971, scenario_seed=3, mechanism=MECHANISMS[0])
        self.assertEqual(a, b)
        self.assertEqual(
            evaluate_potential_outcomes(a.policy, a.evaluator),
            evaluate_potential_outcomes(b.policy, b.evaluator),
        )
        self.assertEqual(a.policy.snapshot.observation.risk_signal.kind, RISK_SIGNAL_KIND)
        self.assertEqual(a.policy.snapshot.observation.risk_signal.horizon, FAILURE_HORIZON)
        self.assertEqual(a.policy.snapshot.observation.risk_signal.source_id, RISK_SOURCE_ID)
        self.assertEqual(a.policy.snapshot.observation.risk_signal.source_version, RISK_SOURCE_VERSION)
        self.assertEqual(a.policy.snapshot.observation.provenance.observed_through_sequence,
                         a.policy.snapshot.observation.provenance.decision_sequence - 1)
        features = extract_features(a.policy.snapshot.observation)
        self.assertEqual(tuple(features.as_ordered_dict()), FEATURE_ORDER)
        self.assertNotIn("susceptibility", vars(a.policy))
        self.assertNotIn("mechanism", vars(a.policy))
        self.assertNotIn("evaluator", vars(a.policy))

    def test_random_streams_are_domain_separated(self):
        self.assertNotEqual(
            _stream_seed(7, "same-scenario", "sensor_noise"),
            _stream_seed(7, "same-scenario", "continuation_execution"),
        )
        self.assertNotEqual(
            _rng(7, "same-scenario", "sensor_noise").random(),
            _rng(7, "same-scenario", "continuation_execution").random(),
        )

    def test_snapshot_rejects_stale_future_incomplete_and_mismatched_provenance(self):
        with self.assertRaises(InvalidObservation):
            _snapshot(observed=-2)
        with self.assertRaises(InvalidObservation):
            _snapshot(observed=0)
        with self.assertRaises(InvalidEventHistory):
            capture_decision_snapshot(
                evidence=_evidence(), pending_action_type="reversible_write",
                plan_version="p", current_step_index=0, planned_step_count=1,
                ledger=TimedTrajectoryLedger("t", complete_from_start=False),
                evidence_timestamp_tick=0, decision_timestamp_tick=0,
            )
        with self.assertRaises(InvalidObservation):
            _snapshot(evidence=_evidence(source="unregistered-source"))
        with self.assertRaises(InvalidObservation):
            _snapshot(evidence=_evidence(source=RISK_SOURCE_ID, version="old-version"))
        with self.assertRaises(InvalidObservation):
            _snapshot(evidence=_evidence(observed=0))

    def test_snapshot_rejects_future_or_same_tick_events(self):
        event = TrajectoryEvent("future", "trajectory-test", 0,
                                "intervention_applied", "i0", "p0")
        ledger = TimedTrajectoryLedger("trajectory-test").append(TimedEvent(event, 0))
        with self.assertRaises(InvalidEventHistory):
            _snapshot(ledger=ledger, decision_tick=0, observed=0, decision_sequence=1)
        with self.assertRaises(InvalidObservation):
            _snapshot(ledger=ledger, decision_tick=2, observed=-1, decision_sequence=0)

    def test_later_ledger_events_do_not_mutate_earlier_snapshot(self):
        ledger = TimedTrajectoryLedger("trajectory-test")
        earlier = _snapshot(ledger=ledger)
        before = extract_features(earlier.observation)
        extended = append_applied_intervention(
            ledger, intervention_id="i0", plan_version="p1", first_tick=1,
            actually_applied=True,
        )
        after = extract_features(earlier.observation)
        self.assertEqual(before, after)
        self.assertEqual(earlier.event_prefix, ())
        self.assertEqual(len(extended.events), 3)

    def test_sensor_and_outcome_streams_are_separate(self):
        generated = build_scenario(root_seed=23, scenario_seed=8, mechanism=MECHANISMS[1])
        altered_evaluator = dataclasses.replace(
            generated.evaluator,
            susceptibility=1.0 - generated.evaluator.susceptibility,
            continuation_uniform=0.99,
            repair_uniform=0.01,
            harm_uniform=0.99,
        )
        self.assertNotEqual(generated.evaluator, altered_evaluator)
        self.assertEqual(
            extract_features(generated.policy.snapshot.observation),
            extract_features(generated.policy.snapshot.observation),
        )


class Phase97EventAndRunnerTests(unittest.TestCase):
    def test_intervention_event_lifecycle_is_emitted_only_after_application(self):
        policy_scenario = PolicyScenario(
            "scenario-test", "initial-state-hash", _snapshot()
        )
        fake = _FakeProposedPolicy("intervene")
        decision = decide(policy_scenario, "proposed", _validated_adapter(fake))
        ledger = TimedTrajectoryLedger(
            policy_scenario.snapshot.observation.provenance.trajectory_id
        )
        with self.assertRaises(InvalidEventHistory):
            record_applied_intervention(
                decision, ledger, intervention_id="i0", plan_version="p1",
                first_tick=1, actually_applied=False,
            )
        self.assertEqual(ledger.events, ())
        applied = record_applied_intervention(
            decision, ledger, intervention_id="i0", plan_version="p1",
            first_tick=1, actually_applied=True, replan_outcome="failed",
        )
        self.assertEqual(
            tuple(item.event.event_type for item in applied.events),
            ("intervention_applied", "replan_started", "replan_failed"),
        )
        self.assertEqual(extract_features(policy_scenario.snapshot.observation).disruption_history, 0.0)
        second = _snapshot(
            ledger=applied, decision_tick=5,
            evidence=_evidence(observed=2), observed=2, decision_sequence=3,
        )
        self.assertAlmostEqual(extract_features(second.observation).disruption_history, 1 / 3)

    def test_baseline_semantics_and_proposed_single_call_allowlist(self):
        exactly_half = _snapshot(evidence=_evidence(anomaly=0.0, alert=True, errors=1, warnings=1))
        self.assertEqual(extract_features(exactly_half.observation).failure_prob, 0.5)
        exactly_half_scenario = PolicyScenario("half", "init", exactly_half)
        self.assertEqual(decide(exactly_half_scenario, "no_intervention").selected_action, "continue")
        self.assertEqual(decide(exactly_half_scenario, "naive_threshold").selected_action, "continue")

        above_half = _snapshot(evidence=_evidence(anomaly=0.004, alert=True, errors=1, warnings=1))
        self.assertGreater(extract_features(above_half.observation).failure_prob, 0.5)
        self.assertEqual(
            decide(PolicyScenario("above", "init", above_half), "naive_threshold").selected_action,
            "abort",
        )

        scenario = build_scenario(root_seed=1, scenario_seed=2, mechanism=MECHANISMS[0]).policy
        fake = _FakeProposedPolicy("intervene")
        metadata = {
            "artifact_version": PHASE97_ARTIFACT_VERSION,
            "feature_schema": PHASE97_FEATURE_SCHEMA,
            "risk_signal_id": PHASE97_RISK_SIGNAL_ID,
            "feature_names": list(FEATURE_ORDER),
            "class_names": list(OUTCOME_CLASSES),
            "cost_matrix": PHASE97_COST_MATRIX,
            "failure_target": "next_verifier_checkpoint_rejects_without_intervention",
            "failure_horizon": "next_verifier_checkpoint_or_final_verification",
        }
        record = decide(scenario, "proposed", Phase97ProposedPolicyAdapter(fake, metadata))
        self.assertEqual(record.selected_action, "intervene")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(tuple(fake.calls[0]), FEATURE_ORDER)
        self.assertNotIn("scenario_id", fake.calls[0])
        self.assertNotIn("mechanism", fake.calls[0])

    def test_runner_rejects_generated_bundle_and_invalid_policy_outputs(self):
        generated = build_scenario(root_seed=1, scenario_seed=2, mechanism=MECHANISMS[0])
        with self.assertRaises(TypeError):
            decide(generated, "no_intervention")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            decide(generated.policy, "proposed", _FakeProposedPolicy("abort"))
        with self.assertRaises(ValueError):
            decide(generated.policy, "unknown")
        self.assertEqual(POLICY_NAMES, ("no_intervention", "naive_threshold", "proposed"))

    def test_phase92_artifact_metadata_is_not_accepted_as_phase97(self):
        old_metadata = {
            "artifact_version": 2,
            "feature_names": list(FEATURE_ORDER),
            "target": "four_class_outcome",
        }
        with self.assertRaises(ValueError):
            validate_phase97_artifact_metadata(old_metadata)
        expected = {
            "artifact_version": PHASE97_ARTIFACT_VERSION,
            "feature_schema": PHASE97_FEATURE_SCHEMA,
            "risk_signal_id": PHASE97_RISK_SIGNAL_ID,
            "feature_names": list(FEATURE_ORDER),
            "class_names": list(OUTCOME_CLASSES),
            "cost_matrix": PHASE97_COST_MATRIX,
            "failure_target": "next_verifier_checkpoint_rejects_without_intervention",
            "failure_horizon": "next_verifier_checkpoint_or_final_verification",
        }
        validate_phase97_artifact_metadata(expected)
        validate_phase97_artifact_metadata({**expected, "training_data_hash": "fixture-hash"})
        adapter_fake = _FakeProposedPolicy()
        adapter = Phase97ProposedPolicyAdapter(adapter_fake, expected)
        valid_features = {name: 0.5 for name in FEATURE_ORDER}
        self.assertEqual(adapter.decide(valid_features), {"selected_action": "continue"})
        self.assertEqual(len(adapter_fake.calls), 1)
        with self.assertRaises(ValueError):
            adapter.decide({"failure_prob": 0.5})
        with self.assertRaises(ValueError):
            Phase97ProposedPolicyAdapter(_FakeProposedPolicy(), old_metadata)

    def test_expected_cost_arithmetic_and_tie_behavior(self):
        outcomes = (
            {"recovery": 1.0, "unnecessary_intervention": 0.0,
             "ineffective_intervention": 0.0, "intervention_harm": 0.0},
            {"recovery": 0.0, "unnecessary_intervention": 1.0,
             "ineffective_intervention": 0.0, "intervention_harm": 0.0},
            {"recovery": 0.0, "unnecessary_intervention": 0.0,
             "ineffective_intervention": 1.0, "intervention_harm": 0.0},
            {"recovery": 0.0, "unnecessary_intervention": 0.0,
             "ineffective_intervention": 0.0, "intervention_harm": 1.0},
        )
        for outcome, expected_continue, expected_intervene, action in zip(
            outcomes, (3.0, 0.0, 3.0, 0.0), (2.0, 2.0, 5.0, 5.0),
            ("intervene", "continue", "continue", "continue"), strict=True,
        ):
            result = expected_costs(outcome)
            self.assertEqual(result["expected_continue_cost"], expected_continue)
            self.assertEqual(result["expected_intervene_cost"], expected_intervene)
            self.assertEqual(result["selected_action"], action)
        tied = expected_costs({
            "recovery": 2 / 3, "unnecessary_intervention": 1 / 3,
            "ineffective_intervention": 0.0, "intervention_harm": 0.0,
        })
        self.assertAlmostEqual(tied["expected_continue_cost"], 2.0)
        self.assertAlmostEqual(tied["expected_intervene_cost"], 2.0)
        self.assertEqual(tied["selected_action"], "continue")
        with self.assertRaises(ValueError):
            expected_costs({"recovery": 1.0})
        invalid = {"recovery": 0.5, "unnecessary_intervention": 0.5,
                   "ineffective_intervention": 0.0, "intervention_harm": "0"}
        with self.assertRaises(ValueError):
            expected_costs(invalid)  # type: ignore[arg-type]

    def test_new_output_guard_refuses_existing_phase92_paths_without_writing(self):
        with self.assertRaises(ValueError):
            reserve_new_output_path("results/phase9_2_diverse_final", kind="results")
        with self.assertRaises(ValueError):
            reserve_new_output_path("models/phase9_2_diverse_final", kind="models")
        output = reserve_new_output_path("results/phase9_7", kind="results")
        self.assertEqual(output.name, "phase9_7")
        self.assertFalse(output.exists())


class Phase97EvaluatorIsolationTests(unittest.TestCase):
    def test_four_outcome_classes_are_possible_for_each_mechanism(self):
        scenario = build_scenario(root_seed=11, scenario_seed=4, mechanism=MECHANISMS[0]).policy
        for mechanism in MECHANISMS:
            observed = set()
            for outcome_class in OUTCOME_CLASSES:
                with self.subTest(mechanism=mechanism, outcome=outcome_class):
                    evaluator = _outcome_case(scenario.scenario_id, mechanism, outcome_class)
                    result = evaluate_potential_outcomes(scenario, evaluator)
                    self.assertEqual(result.outcome_class, outcome_class)
                    self.assertEqual(result.continue_realized_cost, CONTINUE_COSTS[outcome_class])
                    self.assertEqual(result.intervene_realized_cost, INTERVENE_COSTS[outcome_class])
                    self.assertTrue(result.evaluator_only)
                    observed.add(result.outcome_class)
            self.assertEqual(observed, set(OUTCOME_CLASSES))

    def test_evaluator_changes_do_not_change_features_and_scenario_mismatch_fails(self):
        generated = build_scenario(root_seed=9, scenario_seed=5, mechanism=MECHANISMS[2])
        before = extract_features(generated.policy.snapshot.observation)
        changed = dataclasses.replace(
            generated.evaluator,
            susceptibility=0.0,
            continuation_uniform=0.0,
            repair_uniform=1.0,
            harm_uniform=0.0,
        )
        _ = evaluate_potential_outcomes(generated.policy, changed)
        after = extract_features(generated.policy.snapshot.observation)
        self.assertEqual(before, after)
        with self.assertRaises(InvalidScenario):
            evaluate_potential_outcomes(
                generated.policy,
                dataclasses.replace(changed, scenario_id="different-scenario"),
            )

    def test_costs_and_schema_versions_remain_frozen(self):
        self.assertEqual(FAILURE_PENALTY, 3.0)
        self.assertEqual(INTERVENTION_OVERHEAD_COST, 2.0)
        self.assertEqual(CONTINUE_COSTS, phase92.CONTINUE_COSTS)
        self.assertEqual(INTERVENE_COSTS, phase92.INTERVENE_COSTS)
        self.assertEqual(PHASE97_COST_MATRIX["continue"], CONTINUE_COSTS)
        self.assertEqual(PHASE97_COST_MATRIX["intervene"], INTERVENE_COSTS)
        self.assertEqual(tuple(CLASS_TO_INDEX), OUTCOME_CLASSES)
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, "phase9-5.1-observation-v1")


if __name__ == "__main__":
    unittest.main()
