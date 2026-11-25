"""Isolated Phase 9.7 decision runner.

The runner accepts only a public ``PolicyScenario`` and passes exactly the
ordered numeric feature mapping to an injected proposed-policy object. It
does not load models, generate evaluator outcomes, write results, or compare
policies. Intervention events are appended only after the caller confirms
that the action was actually applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from experiments.phase9_5_observations import FEATURE_ORDER, extract_features
from experiments.phase9_7_simulator import (
    OUTCOME_CLASSES,
    PolicyScenario,
    TimedTrajectoryLedger,
    append_applied_intervention,
)


POLICY_NAMES = ("no_intervention", "naive_threshold", "proposed")
PHASE97_ARTIFACT_VERSION = "phase9-7-policy-contract-v1"
PHASE97_FEATURE_SCHEMA = "phase9-7-synthetic-prefix-monitor-v1"
PHASE97_RISK_SIGNAL_ID = "synthetic-prefix-monitor-v1:uncalibrated-score-v1"
PHASE97_COST_MATRIX = {
    "continue": {
        "recovery": 3.0,
        "unnecessary_intervention": 0.0,
        "ineffective_intervention": 3.0,
        "intervention_harm": 0.0,
    },
    "intervene": {
        "recovery": 2.0,
        "unnecessary_intervention": 2.0,
        "ineffective_intervention": 5.0,
        "intervention_harm": 5.0,
    },
}


class ProposedPolicyInterface(Protocol):
    def decide(self, features: dict[str, float]) -> Any: ...


class Phase97ProposedPolicyAdapter:
    """Version gate around an injected classifier; it never loads an artifact."""

    def __init__(self, predictor: ProposedPolicyInterface, metadata: dict[str, Any]):
        validate_phase97_artifact_metadata(metadata)
        self._predictor = predictor

    def decide(self, features: dict[str, float]) -> Any:
        if tuple(features) != FEATURE_ORDER:
            raise ValueError("Phase 9.7 policy input has an invalid feature schema or order")
        for value in features.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Phase 9.7 policy features must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("Phase 9.7 policy features must be finite and in [0, 1]")
        return self._predictor.decide(features)


@dataclass(frozen=True)
class DecisionRecord:
    scenario_id: str
    policy: str
    selected_action: str
    features: tuple[tuple[str, float], ...]
    initial_state_hash: str
    decision_sequence: int

    def __post_init__(self) -> None:
        if not self.scenario_id or self.policy not in POLICY_NAMES:
            raise ValueError("decision record scenario or policy is invalid")
        allowed = {"continue", "intervene", "abort"}
        if self.selected_action not in allowed:
            raise ValueError("decision record action is invalid")
        if tuple(name for name, _ in self.features) != FEATURE_ORDER:
            raise ValueError("decision record feature order is invalid")
        if type(self.decision_sequence) is not int or self.decision_sequence < 0:
            raise ValueError("decision sequence is invalid")


def validate_phase97_artifact_metadata(metadata: Any) -> None:
    """Validate public metadata only; deliberately does not load a model."""
    if not isinstance(metadata, dict):
        raise ValueError("Phase 9.7 model metadata must be a dictionary")
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
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("model artifact metadata is not compatible with the Phase 9.7 schema")


def expected_costs(class_probabilities: dict[str, float]) -> dict[str, float | str]:
    """Compute frozen Phase 9 four-class expected costs; ties continue."""
    if tuple(class_probabilities) != OUTCOME_CLASSES:
        raise ValueError("class probabilities must use the frozen Phase 9 class order")
    probabilities: list[float] = []
    for name in OUTCOME_CLASSES:
        value = class_probabilities[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("class probabilities must be numeric")
        value = float(value)
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError("class probabilities must be finite and in [0, 1]")
        probabilities.append(value)
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError("class probabilities must sum to one")
    continue_cost = sum(
        probability * PHASE97_COST_MATRIX["continue"][name]
        for name, probability in zip(OUTCOME_CLASSES, probabilities, strict=True)
    )
    intervene_cost = sum(
        probability * PHASE97_COST_MATRIX["intervene"][name]
        for name, probability in zip(OUTCOME_CLASSES, probabilities, strict=True)
    )
    return {
        "expected_continue_cost": continue_cost,
        "expected_intervene_cost": intervene_cost,
        "selected_action": "intervene" if intervene_cost < continue_cost else "continue",
    }


def decide(
    scenario: PolicyScenario,
    policy_name: str,
    proposed_policy: Phase97ProposedPolicyAdapter | None = None,
) -> DecisionRecord:
    """Make one decision from a public scenario; no evaluator input is accepted."""
    if not isinstance(scenario, PolicyScenario):
        raise TypeError("runner accepts only a public PolicyScenario")
    if policy_name not in POLICY_NAMES:
        raise ValueError("unknown policy")
    vector = extract_features(scenario.snapshot.observation)
    feature_dict = vector.as_ordered_dict()
    if tuple(feature_dict) != FEATURE_ORDER:
        raise AssertionError("Phase 9.7 feature order changed")

    if policy_name == "no_intervention":
        selected_action = "continue"
    elif policy_name == "naive_threshold":
        selected_action = "abort" if vector.failure_prob > 0.5 else "continue"
    else:
        if not isinstance(proposed_policy, Phase97ProposedPolicyAdapter):
            raise ValueError("a schema-validated Phase 9.7 policy adapter is required")
        # Exactly one call; only the allowlisted feature mapping crosses the boundary.
        result = proposed_policy.decide(feature_dict)
        selected_action = result.get("selected_action") if isinstance(result, dict) else result
        if selected_action not in {"continue", "intervene"}:
            raise ValueError("proposed policy returned an invalid action")

    return DecisionRecord(
        scenario_id=scenario.scenario_id,
        policy=policy_name,
        selected_action=selected_action,
        features=tuple(feature_dict.items()),
        initial_state_hash=scenario.initial_state_hash,
        decision_sequence=scenario.snapshot.observation.provenance.decision_sequence,
    )


def record_applied_intervention(
    decision: DecisionRecord,
    ledger: TimedTrajectoryLedger,
    *, intervention_id: str,
    plan_version: str,
    first_tick: int,
    actually_applied: bool,
    replan_outcome: str = "completed",
) -> TimedTrajectoryLedger:
    """Record an applied action and its replan lifecycle; never rolls back."""
    if decision.selected_action != "intervene":
        raise ValueError("cannot emit intervention events for a non-intervention decision")
    if decision.decision_sequence != len(ledger.events):
        raise ValueError("decision does not refer to the current ledger prefix")
    return append_applied_intervention(
        ledger,
        intervention_id=intervention_id,
        plan_version=plan_version,
        first_tick=first_tick,
        actually_applied=actually_applied,
        replan_outcome=replan_outcome,
    )
