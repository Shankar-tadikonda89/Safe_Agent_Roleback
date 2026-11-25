"""Phase 9.8 matched pilot decision/evaluation primitives.

Decision creation accepts public Phase 9.7 scenarios only. Evaluator potential
outcomes are accepted by a separate post-decision scoring function. This module
does not create scenarios or outcomes and has no CLI that can start a pilot.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.phase9_5_observations import FEATURE_ORDER, extract_features
from experiments.phase9_7_simulator import (
    FAILURE_PENALTY,
    OUTCOME_CLASSES,
    PolicyScenario,
    PotentialOutcomes,
    TimedTrajectoryLedger,
    append_applied_intervention,
)
from experiments.phase9_8_adapter import Phase98ProposedPolicyAdapter
from experiments.phase9_8_prefix_forecast import (
    NativePrefixForecastContext,
    PROVIDER_VERSION as NATIVE_PREFIX_POLICY,
    forecast as run_native_prefix_forecast,
)


PILOT_RUNNER_VERSION = "phase9-8-pilot-runner-v1"
NATIVE_DECISION_SCHEMA_VERSION = "phase9-8-native-prefix-decision-v3"
PREFIX_POLICY_ALIAS = "prefix_branching_approximation"
LEGACY_COMPARATIVE_POLICIES = (
    "no_intervention", "naive_threshold", "proposed", "prefix_branching_approximation",
)
COMPARATIVE_POLICIES = (
    "no_intervention", "naive_threshold", "proposed",
    NATIVE_PREFIX_POLICY,
)
ALL_FROZEN_POLICIES = COMPARATIVE_POLICIES + ("pilot_test",)
PILOT_ROOT_SEED = 20269804
SCENARIOS_PER_GROUP = 256
PILOT_GROUP_COUNT = 10
EXPECTED_PILOT_SCENARIOS = 2560
PILOT_OUTPUT_ROOT = Path("results/phase9_8_pilot")


@dataclass(frozen=True)
class PilotScenario:
    """Policy-visible scenario plus frozen design identifiers, never evaluator truth."""
    policy_scenario: PolicyScenario
    configuration_group_id: str
    mechanism: str
    root_seed: int
    scenario_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.policy_scenario, PolicyScenario):
            raise TypeError("pilot scenario requires a public PolicyScenario")
        if not self.configuration_group_id or not self.mechanism:
            raise ValueError("pilot scenario configuration provenance is required")
        for value in (self.root_seed, self.scenario_seed):
            if type(value) is not int or value < 0:
                raise ValueError("pilot seeds must be non-negative integers")


@dataclass(frozen=True)
class PilotDecision:
    scenario_id: str
    configuration_group_id: str
    mechanism: str
    root_seed: int
    scenario_seed: int
    initial_state_hash: str
    policy: str
    selected_action: str
    decision_sequence: int
    features: tuple[tuple[str, float], ...]
    diagnostics: tuple[tuple[str, Any], ...] = ()
    local_decision_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.policy == PREFIX_POLICY_ALIAS:
            # Additive compatibility for older callers; persisted records name
            # the actual adopted comparator, never the obsolete Phase 9.2 port.
            object.__setattr__(self, "policy", NATIVE_PREFIX_POLICY)
        if self.policy not in COMPARATIVE_POLICIES:
            raise ValueError("pilot decision is not a comparative policy")
        if self.selected_action not in {"continue", "intervene", "abort"}:
            raise ValueError("pilot decision action is invalid")
        if type(self.decision_sequence) is not int or self.decision_sequence < 0:
            raise ValueError("pilot decision sequence is invalid")
        if tuple(name for name, _ in self.features) != FEATURE_ORDER:
            raise ValueError("pilot decision feature order is invalid")
        if not math.isfinite(self.local_decision_latency_seconds) or self.local_decision_latency_seconds < 0:
            raise ValueError("local decision latency is invalid")

    def to_record(self) -> dict[str, Any]:
        # Keep historical records on their original schema; the native policy
        # adds forecast diagnostics under a separately versioned record schema.
        record = {
            "schema_version": (NATIVE_DECISION_SCHEMA_VERSION
                               if self.policy == NATIVE_PREFIX_POLICY else PILOT_RUNNER_VERSION),
            "scenario_id": self.scenario_id,
            "configuration_group_id": self.configuration_group_id,
            "mechanism": self.mechanism,
            "root_seed": self.root_seed,
            "scenario_seed": self.scenario_seed,
            "initial_state_hash": self.initial_state_hash,
            "policy": self.policy,
            "selected_action": self.selected_action,
            "decision_sequence": self.decision_sequence,
            "feature_names": list(FEATURE_ORDER),
            "features": dict(self.features),
            "decision_diagnostics": dict(self.diagnostics),
            "local_decision_latency_seconds": self.local_decision_latency_seconds,
            "runtime_kind": "local_offline_simulation",
        }
        if self.policy == NATIVE_PREFIX_POLICY:
            record["comparator_id"] = NATIVE_PREFIX_POLICY
        return record


def _validated_features(scenario: PolicyScenario) -> dict[str, float]:
    if not isinstance(scenario, PolicyScenario):
        raise TypeError("decision input must be a public PolicyScenario")
    feature_row = extract_features(scenario.snapshot.observation).as_ordered_dict()
    if tuple(feature_row) != FEATURE_ORDER:
        raise ValueError("Phase 9.8 feature order differs from the frozen schema")
    return feature_row


class Phase98PilotRunner:
    """Runs policy decisions over a supplied matched public scenario list."""

    def decide_one(
        self, scenario: PilotScenario, policy: str, *,
        proposed: Phase98ProposedPolicyAdapter | None = None,
        native_prefix_context: NativePrefixForecastContext | None = None,
    ) -> PilotDecision:
        if not isinstance(scenario, PilotScenario):
            raise TypeError("runner accepts PilotScenario public records only")
        if policy == PREFIX_POLICY_ALIAS:
            policy = NATIVE_PREFIX_POLICY
        if policy not in COMPARATIVE_POLICIES:
            raise ValueError("unsupported comparative policy")
        features = _validated_features(scenario.policy_scenario)
        diagnostics: dict[str, Any] = {}
        started = time.perf_counter()
        if policy == "no_intervention":
            action = "continue"
        elif policy == "naive_threshold":
            action = "abort" if features["failure_prob"] > 0.5 else "continue"
        elif policy == "proposed":
            if not isinstance(proposed, Phase98ProposedPolicyAdapter):
                raise ValueError("proposed policy requires the separately validated Phase 9.8 adapter")
            prediction = proposed.decide(features)
            action = prediction["selected_action"]
            diagnostics = prediction
        else:
            if not isinstance(native_prefix_context, NativePrefixForecastContext):
                raise ValueError("native prefix policy requires a validated Phase 9.8 forecast context")
            expected_features = (
                features["reversibility_proxy"], features["trajectory_position"],
                features["disruption_history"],
            )
            actual_features = (
                native_prefix_context.reversibility_proxy,
                native_prefix_context.trajectory_position,
                native_prefix_context.disruption_history,
            )
            if (native_prefix_context.scenario_id != scenario.policy_scenario.scenario_id
                    or native_prefix_context.configuration_group_id != scenario.configuration_group_id
                    or native_prefix_context.root_seed != scenario.root_seed
                    or native_prefix_context.mechanism != scenario.mechanism
                    or actual_features != expected_features):
                raise ValueError("native prefix context does not match decision-time scenario provenance")
            result = run_native_prefix_forecast(native_prefix_context)
            action = result.selected_action
            diagnostics = result.to_diagnostics()
        elapsed = time.perf_counter() - started
        return PilotDecision(
            scenario_id=scenario.policy_scenario.scenario_id,
            configuration_group_id=scenario.configuration_group_id,
            mechanism=scenario.mechanism,
            root_seed=scenario.root_seed,
            scenario_seed=scenario.scenario_seed,
            initial_state_hash=scenario.policy_scenario.initial_state_hash,
            policy=policy,
            selected_action=action,
            decision_sequence=scenario.policy_scenario.snapshot.observation.provenance.decision_sequence,
            features=tuple((name, float(value)) for name, value in features.items()),
            diagnostics=tuple(diagnostics.items()),
            local_decision_latency_seconds=elapsed,
        )

    def decide_matched(
        self, scenarios: Sequence[PilotScenario], *,
        proposed: Phase98ProposedPolicyAdapter,
        native_prefix_contexts: Mapping[str, NativePrefixForecastContext],
    ) -> list[PilotDecision]:
        """Make all comparative decisions from the exact same public scenarios.

        This method intentionally accepts no evaluator records and does not
        generate outcomes. Scenario identity, seeds and initial-state hashes
        are copied to each policy record for later matched-join validation.
        """
        rows = list(scenarios)
        if not rows:
            raise ValueError("matched pilot scenarios cannot be empty")
        ids = [row.policy_scenario.scenario_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("matched pilot scenarios contain duplicate scenario IDs")
        if not isinstance(native_prefix_contexts, Mapping) or set(native_prefix_contexts) != set(ids):
            raise ValueError("native prefix contexts must exactly match the matched scenario IDs")
        policies = COMPARATIVE_POLICIES
        decisions = [
            self.decide_one(
                row, policy, proposed=proposed,
                native_prefix_context=native_prefix_contexts[row.policy_scenario.scenario_id]
                if policy == NATIVE_PREFIX_POLICY else None,
            )
            for row in rows for policy in policies
        ]
        validate_matched_decisions(decisions, rows, expected_policies=policies)
        return decisions


def validate_matched_decisions(
    decisions: Sequence[PilotDecision], scenarios: Sequence[PilotScenario], *,
    expected_policies: Sequence[str] | None = None,
) -> None:
    expected_ids = {row.policy_scenario.scenario_id for row in scenarios}
    if not expected_ids:
        raise ValueError("matched scenario set cannot be empty")
    seen: dict[str, list[PilotDecision]] = {scenario_id: [] for scenario_id in expected_ids}
    for decision in decisions:
        if decision.scenario_id not in seen:
            raise ValueError("decision references a scenario outside the matched set")
        seen[decision.scenario_id].append(decision)
    inferred_sets = [frozenset(row.policy for row in seen[scenario_id]) for scenario_id in expected_ids]
    selected_policies = frozenset(expected_policies) if expected_policies is not None else (inferred_sets[0] if inferred_sets else frozenset())
    if selected_policies not in {frozenset(LEGACY_COMPARATIVE_POLICIES), frozenset(COMPARATIVE_POLICIES)}:
        raise ValueError("matched decisions do not use a supported complete policy set")
    if any(policy_set != selected_policies for policy_set in inferred_sets):
        raise ValueError("matched scenarios have inconsistent policy sets")
    for scenario in scenarios:
        rows = seen[scenario.policy_scenario.scenario_id]
        if len(rows) != len(selected_policies) or {row.policy for row in rows} != set(selected_policies):
            raise ValueError("each scenario must have one decision for every selected comparative policy")
        if len({(row.configuration_group_id, row.mechanism, row.root_seed,
                 row.scenario_seed, row.initial_state_hash) for row in rows}) != 1:
            raise ValueError("matched policy decisions do not share identical scenario provenance")


def append_intervention_events(
    decision: PilotDecision, ledger: TimedTrajectoryLedger, *,
    intervention_id: str, plan_version: str, first_tick: int,
    actually_applied: bool, replan_outcome: str = "completed",
) -> TimedTrajectoryLedger:
    """Append lifecycle events only after actual application; no rollback path."""
    if decision.selected_action != "intervene":
        raise ValueError("cannot record intervention lifecycle for a non-intervention decision")
    if decision.decision_sequence != len(ledger.events):
        raise ValueError("decision does not refer to the current trajectory ledger prefix")
    if actually_applied is not True:
        raise ValueError("intervention lifecycle can be recorded only after application")
    return append_applied_intervention(
        ledger, intervention_id=intervention_id, plan_version=plan_version,
        first_tick=first_tick, actually_applied=True, replan_outcome=replan_outcome,
    )


def counterfactual_application_record(selected_action: str) -> dict[str, Any]:
    """Distinguish a simulated choice from an actually confirmed intervention.

    Phase 9.8 pilot branches evaluate counterfactual potential outcomes; they do
    not execute tool actions or replans. The selected action is observable, but
    actual application is not part of this simulator.
    """
    if selected_action not in {"continue", "intervene", "abort"}:
        raise ValueError("selected action is invalid")
    return {
        "intervention_selected": selected_action == "intervene",
        "confirmed_application": False,
        "intervention_applied": None,
        "execution_semantics": "one_decision_counterfactual_simulation",
    }


def score_frozen_decisions(
    decisions: Sequence[PilotDecision],
    evaluator_outcomes: Mapping[str, PotentialOutcomes],
    *, expected_scenarios: Sequence[PilotScenario],
) -> list[dict[str, Any]]:
    """Join evaluator truth only after a complete in-memory decision set validates.

    This is a checked API boundary, not a durable write-once log or transaction.
    A future runner must persist/freeze the returned decision records before
    passing evaluator outcomes here.
    """
    rows = list(decisions)
    if not rows:
        raise ValueError("cannot score an empty decision set")
    scenario_ids = _validate_complete_decision_set(rows, expected_scenarios)
    # No evaluator mapping is inspected until decision completeness and
    # matched provenance have passed.
    if not isinstance(evaluator_outcomes, Mapping):
        raise TypeError("evaluator outcomes must be a scenario-keyed mapping")
    if set(evaluator_outcomes) != scenario_ids:
        raise ValueError("evaluator outcome IDs do not match the complete decision scenario set")
    if any(not isinstance(value, PotentialOutcomes) or not value.evaluator_only
           for value in evaluator_outcomes.values()):
        raise TypeError("evaluator join requires evaluator-only PotentialOutcomes")
    scored: list[dict[str, Any]] = []
    for row in rows:
        potential = evaluator_outcomes[row.scenario_id]
        if potential.outcome_class not in OUTCOME_CLASSES:
            raise ValueError("evaluator potential outcome class is invalid")
        expected_passes = {
            "recovery": (False, True),
            "unnecessary_intervention": (True, True),
            "ineffective_intervention": (False, False),
            "intervention_harm": (True, False),
        }[potential.outcome_class]
        if (potential.continue_passed, potential.intervene_passed) != expected_passes:
            raise ValueError("evaluator verifier outcomes disagree with the outcome class")
        if (potential.continue_realized_cost != sim_cost("continue", potential.outcome_class)
                or potential.intervene_realized_cost != sim_cost("intervene", potential.outcome_class)):
            raise ValueError("evaluator realized costs disagree with the frozen cost matrix")
        if row.selected_action == "continue":
            passed, cost, steps = potential.continue_passed, potential.continue_realized_cost, 1
        elif row.selected_action == "intervene":
            passed, cost, steps = potential.intervene_passed, potential.intervene_realized_cost, 2
        else:  # naive abort is an honest failure and incurs the frozen failure penalty.
            passed, cost, steps = False, FAILURE_PENALTY, 0
        scored.append({
            **row.to_record(),
            "verifier_passed": bool(passed),
            "success": bool(passed),
            "realized_task_cost": float(cost),
            "steps_executed": steps,
            **counterfactual_application_record(row.selected_action),
            "outcome_class": potential.outcome_class if row.selected_action == "intervene" else None,
            "_continue_passed": bool(potential.continue_passed),
        })
    return scored


def _validate_complete_decision_set(
    rows: Sequence[PilotDecision], expected_scenarios: Sequence[PilotScenario],
) -> set[str]:
    """Reject partial, duplicated, or mismatched policy decisions pre-join."""
    if any(not isinstance(row, PilotDecision) for row in rows):
        raise TypeError("decision set must contain PilotDecision records")
    scenarios = list(expected_scenarios)
    if not scenarios or any(not isinstance(scenario, PilotScenario) for scenario in scenarios):
        raise ValueError("expected scenarios must be a nonempty PilotScenario sequence")
    expected_scenario_ids = [scenario.policy_scenario.scenario_id for scenario in scenarios]
    if len(expected_scenario_ids) != len(set(expected_scenario_ids)):
        raise ValueError("expected scenarios contain duplicate scenario IDs")
    by_scenario: dict[str, list[PilotDecision]] = {}
    pair_keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.scenario_id, row.policy)
        if key in pair_keys:
            raise ValueError("duplicate policy decision for a scenario")
        pair_keys.add(key)
        by_scenario.setdefault(row.scenario_id, []).append(row)
    if len(pair_keys) != len(rows):
        raise ValueError("duplicate policy decision records")
    expected_ids = set(expected_scenario_ids)
    if set(by_scenario) != expected_ids:
        raise ValueError("decision scenario IDs do not match the expected scenario set")
    policy_set = {row.policy for row in rows}
    if policy_set not in (set(LEGACY_COMPARATIVE_POLICIES), set(COMPARATIVE_POLICIES)):
        raise ValueError("decision set must contain one complete supported policy set")
    expected_by_id = {scenario.policy_scenario.scenario_id: scenario for scenario in scenarios}
    for scenario_id, scenario_rows in by_scenario.items():
        if len(scenario_rows) != len(policy_set) or {row.policy for row in scenario_rows} != policy_set:
            raise ValueError(f"scenario {scenario_id} has an incomplete policy decision set")
        provenance = {
            (row.configuration_group_id, row.mechanism, row.root_seed, row.scenario_seed,
             row.initial_state_hash, row.decision_sequence, row.features)
            for row in scenario_rows
        }
        if len(provenance) != 1:
            raise ValueError(f"scenario {scenario_id} has inconsistent matched decision provenance")
        scenario = expected_by_id[scenario_id]
        expected_provenance = (
            scenario.configuration_group_id, scenario.mechanism, scenario.root_seed,
            scenario.scenario_seed, scenario.policy_scenario.initial_state_hash,
            scenario.policy_scenario.snapshot.observation.provenance.decision_sequence,
            tuple((name, float(value)) for name, value in _validated_features(scenario.policy_scenario).items()),
        )
        if next(iter(provenance)) != expected_provenance:
            raise ValueError(f"scenario {scenario_id} decision provenance differs from expected scenario")
    return expected_ids


def calculate_pilot_metrics(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate metrics from scored logs; denominators are explicit and null-safe."""
    rows = list(scored_rows)
    if not rows:
        raise ValueError("metrics require at least one recorded run")
    by_policy: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        policy = row.get("policy")
        if policy not in COMPARATIVE_POLICIES and policy not in LEGACY_COMPARATIVE_POLICIES:
            raise ValueError("comparative metrics exclude pilot_test and unknown policies")
        if row.get("verifier_passed") is not row.get("success"):
            raise ValueError("success field disagrees with verifier outcome")
        by_policy.setdefault(str(policy), []).append(row)
    selected_policy_set = set(by_policy)
    if selected_policy_set not in (set(LEGACY_COMPARATIVE_POLICIES), set(COMPARATIVE_POLICIES)):
        raise ValueError("metrics require one complete supported comparative policy set")
    scenario_sets = {policy: {row.get("scenario_id") for row in policy_rows}
                     for policy, policy_rows in by_policy.items()}
    if any(None in values for values in scenario_sets.values()) or len({frozenset(v) for v in scenario_sets.values()}) != 1:
        raise ValueError("comparative policy logs do not contain matched scenario sets")
    for scenario_id in next(iter(scenario_sets.values())):
        matched = [row for policy_rows in by_policy.values()
                   for row in policy_rows if row.get("scenario_id") == scenario_id]
        if (len(matched) != len(selected_policy_set)
                or {row.get("policy") for row in matched} != selected_policy_set
                or len({
            (row.get("configuration_group_id"), row.get("mechanism"), row.get("root_seed"),
             row.get("scenario_seed"), row.get("initial_state_hash")) for row in matched
        }) != 1):
            raise ValueError("matched policy logs have missing or inconsistent scenario provenance")
    summaries: dict[str, Any] = {}
    for policy, policy_rows in by_policy.items():
        n = len(policy_rows)
        for row in policy_rows:
            new_counterfactual_schema = "intervention_selected" in row
            if (type(row.get("verifier_passed")) is not bool
                    or type(row.get("success")) is not bool
                    or (new_counterfactual_schema and (
                        type(row.get("intervention_selected")) is not bool
                        or row.get("confirmed_application") is not False
                        or row.get("intervention_applied") is not None
                        or row.get("execution_semantics") != "one_decision_counterfactual_simulation"
                    ))
                    or (not new_counterfactual_schema and type(row.get("intervention_applied")) is not bool)
                    or type(row.get("steps_executed")) is not int or row["steps_executed"] < 0
                    or isinstance(row.get("realized_task_cost"), bool)
                    or not isinstance(row.get("realized_task_cost"), (int, float))
                    or not math.isfinite(float(row["realized_task_cost"]))
                    or isinstance(row.get("local_decision_latency_seconds"), bool)
                    or not isinstance(row.get("local_decision_latency_seconds"), (int, float))
                    or not math.isfinite(float(row["local_decision_latency_seconds"]))):
                raise ValueError("scored log contains invalid outcome, cost, step, or latency data")
            selected = (row.get("intervention_selected") if new_counterfactual_schema
                        else row.get("intervention_applied"))
            if (type(row.get("_continue_passed")) is not bool
                    or selected is not (row.get("selected_action") == "intervene")
                    or (selected and row.get("outcome_class") not in OUTCOME_CLASSES)
                    or (not selected and row.get("outcome_class") is not None)):
                raise ValueError("scored log has inconsistent continuation or intervention outcome fields")
        interventions = [row for row in policy_rows if (
            row.get("intervention_selected") is True if "intervention_selected" in row
            else row.get("intervention_applied") is True
        )]
        continuation_failures = sum(
            row.get("_continue_passed") is False for row in policy_rows
        )
        # _continue_passed is evaluator-only audit context added by the caller
        # after decisions; absent values leave recovery undefined, never guessed.
        recovery_count = sum(row.get("outcome_class") == "recovery" for row in interventions)
        unnecessary_count = sum(row.get("outcome_class") == "unnecessary_intervention" for row in interventions)
        harm_count = sum(row.get("outcome_class") == "intervention_harm" for row in interventions)
        summaries[policy] = {
            "attempts": n,
            "success_count": sum(row["verifier_passed"] is True for row in policy_rows),
            "task_success_rate": sum(row["verifier_passed"] is True for row in policy_rows) / n,
            "intervention_count": len(interventions),
            "intervention_rate": len(interventions) / n,
            "confirmed_application_count": (
                sum(row.get("confirmed_application") is True for row in policy_rows)
                if all("intervention_selected" in row for row in policy_rows) else None
            ),
            "intervention_count_definition": "selected intervene actions; this pilot does not execute interventions",
            "disruption_rate": len(interventions) / n,
            "recovery_count": recovery_count,
            "recovery_denominator_continue_failures": continuation_failures,
            "recovery_rate": recovery_count / continuation_failures if continuation_failures else None,
            "unnecessary_count": unnecessary_count,
            "unnecessary_denominator_continue_passes": sum(row.get("_continue_passed") is True for row in policy_rows),
            "harm_count": harm_count,
            "harm_denominator_interventions": len(interventions),
            "harm_rate": harm_count / len(interventions) if interventions else None,
            "mean_realized_task_cost": statistics.mean(float(row["realized_task_cost"]) for row in policy_rows),
            "mean_steps": statistics.mean(int(row["steps_executed"]) for row in policy_rows),
            "mean_local_decision_latency_seconds": statistics.mean(
                float(row["local_decision_latency_seconds"]) for row in policy_rows
            ),
            "latency_interpretation": "local offline runtime, not provider/API latency",
        }
    return {"schema_version": PILOT_RUNNER_VERSION, "policy_metrics": summaries,
            "disruption_rate_definition": "selected intervene actions / all runs; one-decision counterfactual proxy, not confirmed application or measured downstream harm"}


def primary_paired_cost_estimand(
    scored_rows: Sequence[Mapping[str, Any]], *,
    candidate: str = "proposed", baseline: str = "no_intervention",
) -> dict[str, Any]:
    """Return paired candidate-minus-baseline mean costs and frozen cluster CI.

    Per-configuration paired differences are the resampling units. Sampling is
    stratified by the five frozen mechanisms; scenarios within a group are first
    averaged and never treated as independent bootstrap clusters.
    """
    rows = list(scored_rows)
    if candidate not in COMPARATIVE_POLICIES or baseline not in COMPARATIVE_POLICIES or candidate == baseline:
        raise ValueError("paired estimand policies are invalid")
    keyed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("scenario_id")), str(row.get("policy")))
        if key in keyed:
            raise ValueError("duplicate policy result for a matched scenario")
        keyed[key] = row
    by_group: dict[tuple[str, str], list[float]] = {}
    for (scenario_id, policy), candidate_row in keyed.items():
        if policy != candidate:
            continue
        base_row = keyed.get((scenario_id, baseline))
        if base_row is None:
            raise ValueError("paired baseline result is missing")
        for field in ("configuration_group_id", "mechanism", "root_seed", "scenario_seed", "initial_state_hash"):
            if candidate_row.get(field) != base_row.get(field):
                raise ValueError("paired cost records do not share scenario provenance")
        difference = float(candidate_row["realized_task_cost"]) - float(base_row["realized_task_cost"])
        group_key = (str(candidate_row["mechanism"]), str(candidate_row["configuration_group_id"]))
        by_group.setdefault(group_key, []).append(difference)
    if not by_group:
        raise ValueError("paired primary estimand has no matched scenarios")
    group_means = {key: statistics.mean(values) for key, values in by_group.items()}
    observed = statistics.mean(group_means.values())
    by_mechanism: dict[str, list[float]] = {}
    for (mechanism, _group), value in group_means.items():
        by_mechanism.setdefault(mechanism, []).append(value)
    if any(len(values) < 1 for values in by_mechanism.values()):
        raise ValueError("mechanism-stratified group set is invalid")
    rng = random.Random(20269810)
    draws = []
    for _ in range(10_000):
        sampled = [rng.choice(values) for mechanism in sorted(by_mechanism)
                   for values in [by_mechanism[mechanism]] for _ in range(len(values))]
        draws.append(statistics.mean(sampled))
    lower, upper = np.quantile(np.asarray(draws, dtype=np.float64), [0.025, 0.975], method="linear")
    return {
        "candidate": candidate,
        "baseline": baseline,
        "estimand": "mean realized task cost(candidate) - mean realized task cost(baseline), paired by scenario; lower favors candidate",
        "mean_paired_cost_difference": observed,
        "matched_scenarios": sum(len(values) for values in by_group.values()),
        "configuration_groups": len(group_means),
        "bootstrap": {
            "method": "paired mechanism-stratified bootstrap over configuration-group means",
            "resamples": 10_000,
            "seed": 20269810,
            "lower_95_percentile": float(lower),
            "upper_95_percentile": float(upper),
        },
        "interpretation": "exploratory; not a formal power or confirmatory significance claim",
    }


@dataclass(frozen=True)
class PilotTestSelection:
    selected_action: str
    continuation_mean_cost: float
    intervention_mean_cost: float
    scenarios_used: int
    comparative_metric_eligible: bool = False

    @property
    def selected_action_mean_cost(self) -> float:
        return self.continuation_mean_cost if self.selected_action == "continue" else self.intervention_mean_cost

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": PILOT_RUNNER_VERSION,
            "policy": "pilot_test",
            "selected_action": self.selected_action,
            "continuation_mean_cost": self.continuation_mean_cost,
            "intervention_mean_cost": self.intervention_mean_cost,
            "selected_action_mean_cost": self.selected_action_mean_cost,
            "scenarios_used": self.scenarios_used,
            "comparative_metric_eligible": False,
            "cost_is_in_sample_selection_descriptive_only": True,
        }


def select_pilot_test_action(evaluator_outcomes: Mapping[str, PotentialOutcomes]) -> PilotTestSelection:
    """Choose a single action on pilot outcomes for descriptive held-out use.

    This in-sample selection cost is descriptive only and is deliberately not
    passed to comparative metric calculation.
    """
    values = list(evaluator_outcomes.values())
    if not values or any(not isinstance(value, PotentialOutcomes) or not value.evaluator_only for value in values):
        raise ValueError("pilot_test requires nonempty evaluator-only potential outcomes")
    continue_cost = statistics.mean(value.continue_realized_cost for value in values)
    intervene_cost = statistics.mean(value.intervene_realized_cost for value in values)
    return PilotTestSelection(
        selected_action="intervene" if intervene_cost < continue_cost else "continue",
        continuation_mean_cost=continue_cost,
        intervention_mean_cost=intervene_cost,
        scenarios_used=len(values),
    )


def sim_cost(action: str, outcome_class: str) -> float:
    """Read the unchanged Phase 9.7 frozen realized-cost constants."""
    from experiments import phase9_7_simulator as sim

    table = sim.CONTINUE_COSTS if action == "continue" else sim.INTERVENE_COSTS
    return float(table[outcome_class])


def write_jsonl_new(path: Path | str, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write a new JSONL artifact without overwriting an existing path."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite pilot artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    return target
