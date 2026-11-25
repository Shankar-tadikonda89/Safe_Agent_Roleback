from __future__ import annotations

import json
import dataclasses
import unittest
from pathlib import Path

from experiments.phase9_5_observations import FEATURE_ORDER, ObservableRiskEvidence
from experiments.phase9_7_simulator import (
    FAILURE_HORIZON,
    FAILURE_TARGET,
    RISK_SOURCE_ID,
    RISK_SOURCE_VERSION,
    DecisionSnapshot,
    EvaluatorCase,
    PolicyScenario,
    PotentialOutcomes,
    TimedTrajectoryLedger,
    _prefix_digest,
    capture_decision_snapshot,
)
from experiments.phase9_8_adapter import METADATA_PATH, Phase98ProposedPolicyAdapter, _LOAD_GATE_TOKEN
from experiments.phase9_8_prefix_forecast import (
    NativePrefixForecastContext,
    PROVIDER_VERSION as NATIVE_PREFIX_POLICY,
)
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES,
    LEGACY_COMPARATIVE_POLICIES,
    PilotScenario,
    Phase98PilotRunner,
    append_intervention_events,
    calculate_pilot_metrics,
    counterfactual_application_record,
    primary_paired_cost_estimand,
    score_frozen_decisions,
    select_pilot_test_action,
    validate_matched_decisions,
    write_jsonl_new,
)


def _policy_scenario(scenario_id: str, *, anomaly: float = 0.4,
                     alert: bool = False, errors: int = 0, warnings: int = 0) -> PolicyScenario:
    trajectory = f"trajectory-{scenario_id}"
    evidence = ObservableRiskEvidence(
        anomaly_score=anomaly, anomaly_alert=alert,
        visible_error_count=errors, visible_warning_count=warnings,
        source_id=RISK_SOURCE_ID, source_version=RISK_SOURCE_VERSION,
        observed_through_sequence=-1,
    )
    ledger = TimedTrajectoryLedger(trajectory)
    snapshot = capture_decision_snapshot(
        evidence=evidence, pending_action_type="reversible_write",
        plan_version="visible-plan-v1", current_step_index=0,
        planned_step_count=2, ledger=ledger,
        evidence_timestamp_tick=0, decision_timestamp_tick=0,
        observed_through_sequence=-1, decision_sequence=0,
    )
    return PolicyScenario(scenario_id, f"initial-{scenario_id}", snapshot)


class _FakeClassifier:
    def __init__(self, probabilities=(0.1, 0.2, 0.3, 0.4)):
        self.probabilities = probabilities
        scaler_type = type("StandardScaler", (), {})
        classifier_type = type("LogisticRegression", (), {})
        scaler = scaler_type()
        scaler.n_features_in_ = 4
        classifier = classifier_type()
        classifier.classes_ = [0, 1, 2, 3]
        classifier.C = 1.0
        self.named_steps = {"scaler": scaler, "classifier": classifier}
        self.n_features_in_ = 4
        self.calls = 0

    def predict_proba(self, matrix):
        self.calls += 1
        return [self.probabilities]


def _proposed_adapter():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return Phase98ProposedPolicyAdapter(
        _FakeClassifier(), metadata, artifact_sha256=metadata["artifact_sha256"], _gate_token=_LOAD_GATE_TOKEN
    )


def _potential(outcome: str) -> PotentialOutcomes:
    continue_pass = outcome in {"unnecessary_intervention", "intervention_harm"}
    intervene_pass = outcome in {"recovery", "unnecessary_intervention"}
    costs_continue = {"recovery": 3.0, "unnecessary_intervention": 0.0,
                      "ineffective_intervention": 3.0, "intervention_harm": 0.0}
    costs_intervene = {"recovery": 2.0, "unnecessary_intervention": 2.0,
                       "ineffective_intervention": 5.0, "intervention_harm": 5.0}
    return PotentialOutcomes(continue_pass, intervene_pass, outcome,
                             costs_continue[outcome], costs_intervene[outcome])


class Phase98PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = Phase98PilotRunner()
        self.scenarios = [
            PilotScenario(_policy_scenario("scenario-a", anomaly=0.4), "transient_tool_failure-cfg-20",
                          "transient_tool_failure", 20269804, 0),
            PilotScenario(_policy_scenario("scenario-b", anomaly=1.0, alert=True, errors=2, warnings=2), "transient_tool_failure-cfg-20",
                          "transient_tool_failure", 20269804, 1),
        ]

    def _native_contexts(self):
        table = json.loads(Path("results/phase9_8_configuration_table.json").read_text(encoding="utf-8"))
        configuration = next(row for row in table["configurations"]
                             if row["configuration_id"] == "transient_tool_failure-cfg-20")
        return {
            row.policy_scenario.scenario_id: NativePrefixForecastContext.create(
                scenario_id=row.policy_scenario.scenario_id,
                configuration_group_id=row.configuration_group_id,
                root_seed=row.root_seed,
                mechanism=row.mechanism,
                parameters=configuration["parameters"],
                susceptibility=0.4,
                reversibility_proxy=0.65,
                trajectory_position=0.0,
                disruption_history=0.0,
            )
            for row in self.scenarios
        }

    def test_counterfactual_selection_is_not_claimed_as_confirmed_application(self):
        selected = counterfactual_application_record("intervene")
        self.assertTrue(selected["intervention_selected"])
        self.assertFalse(selected["confirmed_application"])
        self.assertIsNone(selected["intervention_applied"])
        self.assertEqual(selected["execution_semantics"], "one_decision_counterfactual_simulation")
        continued = counterfactual_application_record("continue")
        self.assertFalse(continued["intervention_selected"])
        self.assertFalse(continued["confirmed_application"])
        self.assertIsNone(continued["intervention_applied"])

    def test_matched_decisions_have_one_same_scenario_per_comparative_policy(self):
        model = _proposed_adapter()
        decisions = self.runner.decide_matched(
            self.scenarios, proposed=model,
            native_prefix_contexts=self._native_contexts(),
        )
        validate_matched_decisions(decisions, self.scenarios)
        self.assertEqual(len(decisions), len(self.scenarios) * len(COMPARATIVE_POLICIES))
        self.assertEqual(model._model.calls, len(self.scenarios))
        for scenario in self.scenarios:
            rows = [d for d in decisions if d.scenario_id == scenario.policy_scenario.scenario_id]
            self.assertEqual({d.policy for d in rows}, set(COMPARATIVE_POLICIES))
            self.assertEqual(len({d.initial_state_hash for d in rows}), 1)
            native = next(d for d in rows if d.policy == NATIVE_PREFIX_POLICY)
            diagnostics = dict(native.diagnostics)
            self.assertEqual(diagnostics["forecast_count"], 8)
            self.assertEqual(diagnostics["forecast_overhead_units"], 8)
            self.assertNotIn("realized_task_cost", diagnostics)
            self.assertEqual(native.to_record()["schema_version"], "phase9-8-native-prefix-decision-v3")
        legacy = self.runner.decide_one(self.scenarios[0], "no_intervention")
        self.assertEqual(legacy.to_record()["schema_version"], "phase9-8-pilot-runner-v1")

    def test_native_context_is_required_for_the_sole_prefix_comparator(self):
        with self.assertRaises(ValueError):
            self.runner.decide_matched(self.scenarios, proposed=_proposed_adapter(), native_prefix_contexts={})

    def test_native_context_must_match_scenario_identity_and_visible_features(self):
        scenario = self.scenarios[0]
        wrong = NativePrefixForecastContext.create(
            scenario_id="other", configuration_group_id=scenario.configuration_group_id,
            root_seed=scenario.root_seed,
            mechanism=scenario.mechanism,
            parameters=dict(self._native_contexts()[scenario.policy_scenario.scenario_id].parameters),
            susceptibility=0.4, reversibility_proxy=0.65,
            trajectory_position=0.0, disruption_history=0.0,
        )
        with self.assertRaises(ValueError):
            self.runner.decide_one(scenario, NATIVE_PREFIX_POLICY, native_prefix_context=wrong)
        with self.assertRaises(ValueError):
            self.runner.decide_matched(
                self.scenarios, proposed=_proposed_adapter(),
                native_prefix_contexts={"scenario-a": wrong},
            )

    def test_repeated_decision_pass_is_deterministic_excluding_runtime_measurement(self):
        first = self.runner.decide_matched(self.scenarios, proposed=_proposed_adapter(),
                                           native_prefix_contexts=self._native_contexts())
        second = self.runner.decide_matched(self.scenarios, proposed=_proposed_adapter(),
                                            native_prefix_contexts=self._native_contexts())
        stable = lambda rows: [
            (r.scenario_id, r.configuration_group_id, r.mechanism, r.root_seed,
             r.scenario_seed, r.initial_state_hash, r.policy, r.selected_action,
             r.features, tuple((key, value) for key, value in r.diagnostics
                               if key != "forecast_runtime_seconds")) for r in rows
        ]
        self.assertEqual(stable(first), stable(second))

    def test_baseline_semantics_prefix_cost_and_strict_threshold(self):
        model = _proposed_adapter()
        base = self.runner.decide_one(self.scenarios[0], "no_intervention")
        self.assertEqual(base.selected_action, "continue")
        under = self.runner.decide_one(self.scenarios[0], "naive_threshold")
        over = self.runner.decide_one(self.scenarios[1], "naive_threshold")
        self.assertEqual(under.selected_action, "continue")
        self.assertEqual(over.selected_action, "abort")
        prefix = self.runner.decide_one(
            self.scenarios[0], NATIVE_PREFIX_POLICY,
            proposed=model, native_prefix_context=self._native_contexts()["scenario-a"],
        )
        self.assertEqual(prefix.selected_action, "continue")
        self.assertEqual(dict(prefix.diagnostics)["forecast_count"], 8)

    def test_evaluator_join_happens_after_decisions_and_aborts_remain_failures(self):
        decisions = self.runner.decide_matched(
            self.scenarios, proposed=_proposed_adapter(),
            native_prefix_contexts=self._native_contexts(),
        )
        outcomes = {
            "scenario-a": _potential("ineffective_intervention"),
            "scenario-b": _potential("recovery"),
        }
        scored = score_frozen_decisions(decisions, outcomes, expected_scenarios=self.scenarios)
        abort = next(row for row in scored if row["scenario_id"] == "scenario-b" and row["policy"] == "naive_threshold")
        self.assertFalse(abort["success"])
        self.assertEqual(abort["realized_task_cost"], 3.0)
        self.assertFalse(any("evaluator" in key for d in decisions for key, _ in d.diagnostics))

    def test_incomplete_duplicate_and_provenance_mismatched_decisions_reject_before_outcome_access(self):
        from collections.abc import Mapping

        class OutcomeAccessTrap(Mapping):
            def __init__(self):
                self.accesses = 0

            def __iter__(self):
                self.accesses += 1
                return iter(())

            def __len__(self):
                self.accesses += 1
                return 0

            def __getitem__(self, key):
                self.accesses += 1
                raise KeyError(key)

            def values(self):
                self.accesses += 1
                return ()

        decisions = self.runner.decide_matched(
            self.scenarios, proposed=_proposed_adapter(),
            native_prefix_contexts=self._native_contexts(),
        )
        invalid_sets = (
            decisions[:-1],
            decisions + [decisions[0]],
            [dataclasses.replace(decisions[0], initial_state_hash="different")] + decisions[1:],
        )
        for invalid in invalid_sets:
            outcomes = OutcomeAccessTrap()
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                score_frozen_decisions(invalid, outcomes, expected_scenarios=self.scenarios)
            self.assertEqual(outcomes.accesses, 0)
        with self.assertRaises(ValueError):
            score_frozen_decisions(
                decisions, {"scenario-a": _potential("recovery")},
                expected_scenarios=self.scenarios,
            )

    def test_legacy_policy_records_remain_supported_by_metrics(self):
        rows = []
        for scenario in self.scenarios:
            for policy in LEGACY_COMPARATIVE_POLICIES:
                rows.append({
                    "scenario_id": scenario.policy_scenario.scenario_id,
                    "configuration_group_id": scenario.configuration_group_id,
                    "mechanism": scenario.mechanism, "root_seed": scenario.root_seed,
                    "scenario_seed": scenario.scenario_seed,
                    "initial_state_hash": scenario.policy_scenario.initial_state_hash,
                    "policy": policy, "selected_action": "continue",
                    "verifier_passed": True, "success": True,
                    "intervention_applied": False, "steps_executed": 1,
                    "realized_task_cost": 0.0, "local_decision_latency_seconds": 0.0,
                    "outcome_class": None, "_continue_passed": True,
                })
        metrics = calculate_pilot_metrics(rows)
        self.assertEqual(set(metrics["policy_metrics"]), set(LEGACY_COMPARATIVE_POLICIES))
        self.assertEqual(metrics["policy_metrics"]["no_intervention"]["attempts"], 2)

    def test_metrics_denominators_and_pilot_test_exclusion(self):
        decisions = self.runner.decide_matched(
            self.scenarios, proposed=_proposed_adapter(),
            native_prefix_contexts=self._native_contexts(),
        )
        scored = score_frozen_decisions(decisions, {
            "scenario-a": _potential("unnecessary_intervention"),
            "scenario-b": _potential("recovery"),
        }, expected_scenarios=self.scenarios)
        metrics = calculate_pilot_metrics(scored)["policy_metrics"]
        self.assertEqual(metrics["naive_threshold"]["attempts"], 2)
        self.assertEqual(metrics["naive_threshold"]["success_count"], 1)
        self.assertEqual(metrics["no_intervention"]["intervention_count"], 0)
        self.assertIsNone(metrics["no_intervention"]["harm_rate"])
        proposed_index = next(i for i, row in enumerate(decisions)
                              if row.scenario_id == "scenario-b" and row.policy == "proposed")
        decisions[proposed_index] = dataclasses.replace(decisions[proposed_index], selected_action="intervene")
        recovery_scored = score_frozen_decisions(decisions, {
            "scenario-a": _potential("unnecessary_intervention"),
            "scenario-b": _potential("recovery"),
        }, expected_scenarios=self.scenarios)
        proposed_metrics = calculate_pilot_metrics(recovery_scored)["policy_metrics"]["proposed"]
        self.assertEqual(proposed_metrics["intervention_count"], 1)
        self.assertEqual(proposed_metrics["recovery_count"], 1)
        self.assertEqual(proposed_metrics["recovery_denominator_continue_failures"], 1)
        self.assertEqual(proposed_metrics["recovery_rate"], 1.0)
        with self.assertRaises(ValueError):
            calculate_pilot_metrics(scored + [{**scored[0], "policy": "pilot_test"}])

    def test_metrics_reject_invalid_or_unmatched_logs(self):
        decisions = self.runner.decide_matched(
            self.scenarios, proposed=_proposed_adapter(),
            native_prefix_contexts=self._native_contexts(),
        )
        scored = score_frozen_decisions(decisions, {
            "scenario-a": _potential("unnecessary_intervention"),
            "scenario-b": _potential("recovery"),
        }, expected_scenarios=self.scenarios)
        with self.assertRaises(ValueError):
            calculate_pilot_metrics(scored[:-1])
        invalid = [dict(row) for row in scored]
        invalid[0]["realized_task_cost"] = float("nan")
        with self.assertRaises(ValueError):
            calculate_pilot_metrics(invalid)

    def test_primary_paired_cost_estimand_uses_scenario_matched_group_costs(self):
        decisions = self.runner.decide_matched(
            self.scenarios, proposed=_proposed_adapter(),
            native_prefix_contexts=self._native_contexts(),
        )
        scored = score_frozen_decisions(decisions, {
            "scenario-a": _potential("unnecessary_intervention"),
            "scenario-b": _potential("recovery"),
        }, expected_scenarios=self.scenarios)
        result = primary_paired_cost_estimand(scored)
        self.assertEqual(result["matched_scenarios"], 2)
        self.assertEqual(result["configuration_groups"], 1)
        self.assertEqual(result["mean_paired_cost_difference"], 0.0)
        self.assertEqual(result["bootstrap"]["resamples"], 10_000)
        self.assertEqual(primary_paired_cost_estimand(scored), result)

    def test_pilot_action_selection_is_descriptive_and_ties_continue(self):
        outcomes = {"a": _potential("recovery"), "b": _potential("recovery"),
                    "c": _potential("unnecessary_intervention")}
        selection = select_pilot_test_action(outcomes)
        self.assertEqual(selection.selected_action, "continue")
        self.assertFalse(selection.comparative_metric_eligible)
        self.assertEqual(selection.scenarios_used, 3)
        self.assertAlmostEqual(selection.selected_action_mean_cost, 2.0)
        self.assertFalse(selection.to_record()["comparative_metric_eligible"])
        self.assertAlmostEqual(selection.selected_action_mean_cost, 2.0)
        self.assertFalse(selection.to_record()["comparative_metric_eligible"])

    def test_intervention_events_require_actual_application_and_never_rollback(self):
        adapter = _proposed_adapter()
        # Force intervention probabilities making intervention strictly cheaper.
        adapter._model.probabilities = (0.9, 0.05, 0.03, 0.02)
        intervention_scenario = PilotScenario(
            _policy_scenario("scenario-intervene"), "cfg-a", "transient_tool_failure", 20269804, 3
        )
        decision = self.runner.decide_one(intervention_scenario, "proposed", proposed=adapter)
        self.assertEqual(decision.selected_action, "intervene")
        ledger = TimedTrajectoryLedger("trajectory-scenario-intervene")
        with self.assertRaises(ValueError):
            append_intervention_events(
                decision, ledger, intervention_id="i1", plan_version="p1",
                first_tick=1, actually_applied=False,
            )
        applied = append_intervention_events(
            decision, ledger, intervention_id="i1", plan_version="p1",
            first_tick=1, actually_applied=True,
        )
        self.assertEqual(tuple(x.event.event_type for x in applied.events),
                         ("intervention_applied", "replan_started", "replan_completed"))

    def test_log_writer_refuses_overwrite(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "new.jsonl"
            write_jsonl_new(path, [{"record": 1}])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"record": 1})
            with self.assertRaises(FileExistsError):
                write_jsonl_new(path, [{"record": 2}])


if __name__ == "__main__":
    unittest.main()
