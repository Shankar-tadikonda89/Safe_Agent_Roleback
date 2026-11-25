from __future__ import annotations

import dataclasses
import unittest

from experiments import phase9_2_simulator as phase92
from experiments.phase9_5_event_history import (
    InvalidEventHistory,
    TrajectoryEvent,
    TrajectoryEventLedger,
    reduce_disruption_history,
)
from experiments.phase9_5_observations import (
    FAILURE_HORIZON,
    FAILURE_TARGET,
    FEATURE_ORDER,
    OBSERVATION_SCHEMA_VERSION,
    REVERSIBILITY_MAPPING_VERSION,
    REVERSIBILITY_VALUES,
    RISK_SIGNAL_KIND,
    InvalidObservation,
    ObservableRiskEvidence,
    ObservationProvenance,
    build_decision_observation,
    extract_features,
)
from experiments.phase9_5_simulator import (
    CONTINUE_COSTS,
    FAILURE_PENALTY,
    INTERVENTION_OVERHEAD_COST,
    INTERVENE_COSTS,
    MECHANISMS,
    OUTCOME_CLASSES,
    TRANSITIONS,
    InvalidSimulatorFixture,
    EvaluatorTruth,
    deterministic_outcome_fixture,
    evaluate_potential_outcomes,
)


def _event(trajectory: str, sequence: int, event_type: str, intervention: str) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id=f"event-{sequence}", trajectory_id=trajectory, sequence=sequence,
        event_type=event_type, intervention_id=intervention, plan_version=f"plan-{sequence}",
    )


def _observation(*, decision_sequence: int = 0, observed_sequence: int = -1,
                 ledger: TrajectoryEventLedger | None = None, action: str = "reversible_write",
                 current: int = 0, planned: int = 2, plan: str = "plan-v1"):
    trajectory = "trajectory-test"
    evidence = ObservableRiskEvidence(
        anomaly_score=0.4, anomaly_alert=False, visible_error_count=1,
        visible_warning_count=0, source_id="visible-fixture", source_version="v1",
        observed_through_sequence=observed_sequence,
    )
    return build_decision_observation(
        evidence=evidence, pending_action_type=action, plan_version=plan,
        current_step_index=current, planned_step_count=planned,
        ledger=ledger or TrajectoryEventLedger(trajectory, True),
        provenance=ObservationProvenance(
            trajectory_id=trajectory, observation_id="snapshot-1",
            decision_sequence=decision_sequence, observed_through_sequence=observed_sequence,
            plan_source_id="plan-source-v1", action_metadata_source_id="action-source-v1",
        ),
    )


class Phase95ObservationContractTests(unittest.TestCase):
    def test_locked_feature_order_score_label_and_provenance(self):
        observation = _observation()
        vector = extract_features(observation)
        self.assertEqual(FEATURE_ORDER, (
            "failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history",
        ))
        self.assertEqual(tuple(vector.as_ordered_dict()), FEATURE_ORDER)
        self.assertEqual(vector.schema_version, OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(vector.failure_signal_kind, RISK_SIGNAL_KIND)
        self.assertEqual(observation.risk_signal.target, FAILURE_TARGET)
        self.assertEqual(observation.risk_signal.horizon, FAILURE_HORIZON)
        # 0.25 * anomaly + 0.25 * alert + 0.25 * capped errors + 0.25 * capped warnings
        self.assertAlmostEqual(vector.failure_prob, 0.225)

    def test_risk_score_rejects_invalid_numeric_types_and_ranges(self):
        for invalid in (True, "0.5", float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InvalidObservation):
                    ObservableRiskEvidence(
                        anomaly_score=invalid, anomaly_alert=False, visible_error_count=0,
                        visible_warning_count=0, source_id="s", source_version="v",
                        observed_through_sequence=-1,
                    )

    def test_missing_unknown_action_and_invalid_plan_are_rejected(self):
        for action in (None, "unknown"):
            with self.subTest(action=action), self.assertRaises(InvalidObservation):
                _observation(action=action)
        with self.assertRaises(InvalidObservation):
            _observation(planned=0)
        with self.assertRaises(InvalidObservation):
            _observation(current=2, planned=2)
        with self.assertRaises(InvalidObservation):
            _observation(current=-1, planned=2)

    def test_fixed_versioned_reversibility_mapping(self):
        self.assertEqual(REVERSIBILITY_MAPPING_VERSION, "action-reversibility-v1")
        self.assertEqual(REVERSIBILITY_VALUES, {
            "read_only": 0.95, "reversible_write": 0.65, "irreversible_write": 0.20,
        })
        for action, expected in REVERSIBILITY_VALUES.items():
            self.assertEqual(extract_features(_observation(action=action)).reversibility_proxy, expected)

    def test_visible_plan_position_edge_cases_and_plan_version_snapshot(self):
        self.assertEqual(extract_features(_observation(current=0, planned=1)).trajectory_position, 0.0)
        self.assertEqual(extract_features(_observation(current=0, planned=5)).trajectory_position, 0.0)
        self.assertEqual(extract_features(_observation(current=4, planned=5)).trajectory_position, 1.0)
        old = _observation(current=1, planned=3, plan="plan-old")
        old_value = extract_features(old).trajectory_position
        updated = dataclasses.replace(old, plan_version="plan-new", planned_step_count=5)
        self.assertEqual(old.plan_version, "plan-old")
        self.assertEqual(extract_features(old).trajectory_position, old_value)
        self.assertEqual(extract_features(updated).trajectory_position, 0.25)

    def test_feature_extraction_is_independent_of_evaluator_truth_and_past_outcomes(self):
        a = deterministic_outcome_fixture("transient_tool_failure", "recovery")
        b = deterministic_outcome_fixture("transient_tool_failure", "intervention_harm")
        self.assertNotEqual(
            evaluate_potential_outcomes(a.observation, a.evaluator_truth),
            evaluate_potential_outcomes(b.observation, b.evaluator_truth),
        )
        self.assertEqual(extract_features(a.observation), extract_features(b.observation))
        with self.assertRaises(InvalidObservation):
            extract_features(a.evaluator_truth)  # type: ignore[arg-type]

    def test_observation_and_feature_vector_are_immutable(self):
        observation = _observation()
        vector = extract_features(observation)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.plan_version = "rewritten"  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            vector.failure_prob = 0.99  # type: ignore[misc]

    def test_provenance_must_precede_decision(self):
        with self.assertRaises(InvalidObservation):
            ObservationProvenance(
                trajectory_id="t", observation_id="s", decision_sequence=2,
                observed_through_sequence=2, plan_source_id="p", action_metadata_source_id="a",
            )


class Phase95EventHistoryTests(unittest.TestCase):
    def test_complete_empty_history_is_zero_and_append_is_immutable(self):
        ledger = TrajectoryEventLedger("t", True)
        reduced = reduce_disruption_history(ledger, 0)
        self.assertEqual(reduced.value, 0.0)
        extended = ledger.append(_event("t", 0, "intervention_applied", "i1"))
        self.assertEqual(ledger.events, ())
        self.assertEqual(len(extended.events), 1)
        self.assertAlmostEqual(reduce_disruption_history(extended, 1).value, 1 / 3)

    def test_only_prior_applied_interventions_count_and_normalization_saturates(self):
        ledger = TrajectoryEventLedger("t", True)
        for i in range(4):
            ledger = ledger.append(_event("t", len(ledger.events), "intervention_applied", f"i{i}"))
            ledger = ledger.append(_event("t", len(ledger.events), "replan_started", f"i{i}"))
            ledger = ledger.append(_event("t", len(ledger.events), "replan_completed", f"i{i}"))
        history = reduce_disruption_history(ledger, 12)
        self.assertEqual(history.applied_intervention_count, 4)
        self.assertEqual(history.value, 1.0)
        self.assertEqual(len(history.event_ids), 4)

    def test_future_events_do_not_change_prefix_reduction(self):
        prefix = TrajectoryEventLedger("t", True)
        prefix = prefix.append(_event("t", 0, "intervention_applied", "i1"))
        prefix = prefix.append(_event("t", 1, "replan_started", "i1"))
        prefix = prefix.append(_event("t", 2, "replan_completed", "i1"))
        with_future = prefix.append(_event("t", 3, "intervention_applied", "future"))
        a = reduce_disruption_history(prefix, 3)
        b = reduce_disruption_history(with_future, 3)
        self.assertEqual(a, b)
        self.assertEqual(a.value, 1 / 3)

    def test_incomplete_gapped_duplicate_and_cross_trajectory_histories_fail(self):
        with self.assertRaises(InvalidEventHistory):
            reduce_disruption_history(TrajectoryEventLedger("t", False), 0)
        gapped = TrajectoryEventLedger("t", True, (_event("t", 1, "intervention_applied", "i"),))
        with self.assertRaises(InvalidEventHistory):
            reduce_disruption_history(gapped, 1)
        duplicate = _event("t", 0, "intervention_applied", "i")
        repeated = TrajectoryEventLedger("t", True, (duplicate, dataclasses.replace(duplicate, sequence=1)))
        with self.assertRaises(InvalidEventHistory):
            reduce_disruption_history(repeated, 2)
        with self.assertRaises(InvalidEventHistory):
            TrajectoryEventLedger("t", True).append(_event("other", 0, "intervention_applied", "i"))

    def test_replan_lifecycle_must_be_consistent(self):
        start_without_apply = TrajectoryEventLedger(
            "t", True, (_event("t", 0, "replan_started", "i"),),
        )
        with self.assertRaises(InvalidEventHistory):
            reduce_disruption_history(start_without_apply, 1)
        dangling = TrajectoryEventLedger("t", True)
        dangling = dangling.append(_event("t", 0, "intervention_applied", "i"))
        dangling = dangling.append(_event("t", 1, "replan_started", "i"))
        with self.assertRaises(InvalidEventHistory):
            reduce_disruption_history(dangling, 2)

    def test_observation_rejects_history_from_another_trajectory_and_accepts_valid_prior_prefix(self):
        wrong_trajectory = TrajectoryEventLedger("other", True)
        with self.assertRaises(InvalidObservation):
            _observation(ledger=wrong_trajectory)
        prior = TrajectoryEventLedger("trajectory-test", True)
        prior = prior.append(_event("trajectory-test", 0, "intervention_applied", "i1"))
        observation = _observation(decision_sequence=1, observed_sequence=0, ledger=prior)
        self.assertAlmostEqual(extract_features(observation).disruption_history, 1 / 3)


class Phase95SimulatorTests(unittest.TestCase):
    def test_each_mechanism_has_deterministic_fixtures_for_all_outcome_classes(self):
        self.assertEqual(len(TRANSITIONS), len(MECHANISMS))
        self.assertEqual(set(TRANSITIONS), set(MECHANISMS))
        for mechanism in MECHANISMS:
            observed = set()
            for outcome_class in OUTCOME_CLASSES:
                with self.subTest(mechanism=mechanism, outcome=outcome_class):
                    fixture = deterministic_outcome_fixture(mechanism, outcome_class)
                    first = evaluate_potential_outcomes(fixture.observation, fixture.evaluator_truth)
                    second = evaluate_potential_outcomes(fixture.observation, fixture.evaluator_truth)
                    self.assertEqual(first, second)
                    self.assertEqual(first.outcome_class, outcome_class)
                    self.assertEqual(first.continue_realized_cost, CONTINUE_COSTS[outcome_class])
                    self.assertEqual(first.intervene_realized_cost, INTERVENE_COSTS[outcome_class])
                    self.assertTrue(first.evaluator_only)
                    observed.add(first.outcome_class)
            self.assertEqual(observed, set(OUTCOME_CLASSES))

    def test_phase_9_cost_table_is_unchanged(self):
        self.assertEqual(CONTINUE_COSTS, phase92.CONTINUE_COSTS)
        self.assertEqual(INTERVENE_COSTS, phase92.INTERVENE_COSTS)
        self.assertEqual(FAILURE_PENALTY, phase92.FAILURE_PENALTY)
        self.assertEqual(INTERVENTION_OVERHEAD_COST, phase92.INTERVENTION_OVERHEAD_COST)

    def test_irreversible_mechanism_rejects_attempted_rollback(self):
        fixture = deterministic_outcome_fixture("irreversible_action", "recovery")
        with self.assertRaises(InvalidSimulatorFixture):
            dataclasses.replace(fixture.evaluator_truth, action_already_completed=True)

    def test_fixture_observations_are_identical_across_evaluator_categories(self):
        observations = [
            deterministic_outcome_fixture("stale_contradictory_evidence", outcome).observation
            for outcome in OUTCOME_CLASSES
        ]
        self.assertTrue(all(observation == observations[0] for observation in observations))
        vectors = [extract_features(observation) for observation in observations]
        self.assertTrue(all(vector == vectors[0] for vector in vectors))


if __name__ == "__main__":
    unittest.main()
