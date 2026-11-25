"""Versioned Phase 9.5 deterministic mechanism fixtures.

Decision observations are public, immutable prefix inputs. ``EvaluatorTruth``
and ``PotentialOutcomes`` are evaluator-only types and are never attached to an
observation. These fixtures validate transition semantics; they are not an
evaluation dataset and imply no real-world effectiveness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from experiments.phase9_5_observations import (
    DecisionObservation,
    ObservableRiskEvidence,
    ObservationProvenance,
    build_decision_observation,
)
from experiments.phase9_5_event_history import TrajectoryEventLedger


SIMULATOR_VERSION = "phase9-5.1-observation-validity-v1"
OUTCOME_CLASSES = (
    "recovery", "unnecessary_intervention", "ineffective_intervention", "intervention_harm",
)
MECHANISMS = (
    "transient_tool_failure",
    "stale_contradictory_evidence",
    "irreversible_action",
    "late_trajectory_failure",
    "replanning_disruption",
)

# Frozen Phase 9 costs, copied verbatim and covered by compatibility tests.
FAILURE_PENALTY = 3.0
INTERVENTION_OVERHEAD_COST = 2.0
CONTINUE_COSTS = {
    "recovery": 3.0,
    "unnecessary_intervention": 0.0,
    "ineffective_intervention": 3.0,
    "intervention_harm": 0.0,
}
INTERVENE_COSTS = {
    "recovery": 2.0,
    "unnecessary_intervention": 2.0,
    "ineffective_intervention": 5.0,
    "intervention_harm": 5.0,
}


class InvalidSimulatorFixture(ValueError):
    """Raised for inconsistent evaluator-only mechanism inputs."""


@dataclass(frozen=True)
class EvaluatorTruth:
    """Hidden simulator state. Never place this object in a policy observation."""
    mechanism: str
    base_action_passes: bool = True
    fault_present: bool = False
    repair_succeeds: bool = False
    stale_evidence: bool = False
    contradictory_evidence: bool = False
    evidence_refresh_succeeds: bool = False
    pending_action_would_pass: bool = True
    guarded_replacement_passes: bool = False
    continuation_budget_ok: bool = True
    intervention_budget_ok: bool = True
    plan_inconsistent: bool = False
    corrective_replan_succeeds: bool = False
    intervention_harm: bool = False
    replan_harm: bool = False
    action_already_completed: bool = False

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise InvalidSimulatorFixture("unknown mechanism")
        for name in (
            "base_action_passes", "fault_present", "repair_succeeds", "stale_evidence",
            "contradictory_evidence", "evidence_refresh_succeeds", "pending_action_would_pass",
            "guarded_replacement_passes", "continuation_budget_ok", "intervention_budget_ok",
            "plan_inconsistent", "corrective_replan_succeeds", "intervention_harm",
            "replan_harm", "action_already_completed",
        ):
            if type(getattr(self, name)) is not bool:
                raise InvalidSimulatorFixture(f"{name} must be boolean")
        if self.mechanism == "irreversible_action" and self.action_already_completed:
            raise InvalidSimulatorFixture("intervention cannot reverse an already completed action")


@dataclass(frozen=True)
class PotentialOutcomes:
    """Evaluator-only paired outcomes and frozen realized costs."""
    continue_passed: bool
    intervene_passed: bool
    outcome_class: str
    continue_realized_cost: float
    intervene_realized_cost: float
    evaluator_only: bool = True


@dataclass(frozen=True)
class MechanismFixture:
    observation: DecisionObservation
    evaluator_truth: EvaluatorTruth


def _transition_transient(t: EvaluatorTruth) -> tuple[bool, bool]:
    continued = t.base_action_passes and not t.fault_present
    intervened = t.base_action_passes and (not t.fault_present or t.repair_succeeds)
    return continued, intervened and not t.intervention_harm


def _transition_stale_evidence(t: EvaluatorTruth) -> tuple[bool, bool]:
    evidence_problem = t.stale_evidence or t.contradictory_evidence
    continued = t.base_action_passes and not evidence_problem
    resolved = not evidence_problem or t.evidence_refresh_succeeds
    return continued, t.base_action_passes and resolved and not t.intervention_harm


def _transition_irreversible(t: EvaluatorTruth) -> tuple[bool, bool]:
    # The pending action may be stopped/replaced before execution; completed work
    # is never reversed. The replacement has its own evaluator-side outcome.
    continued = t.pending_action_would_pass
    intervened = t.guarded_replacement_passes and not t.intervention_harm
    return continued, intervened


def _transition_late(t: EvaluatorTruth) -> tuple[bool, bool]:
    continued = t.base_action_passes and t.continuation_budget_ok
    repaired_or_continued = t.base_action_passes or t.repair_succeeds
    intervened = t.intervention_budget_ok and repaired_or_continued and not t.intervention_harm
    return continued, intervened


def _transition_replanning(t: EvaluatorTruth) -> tuple[bool, bool]:
    continued = t.base_action_passes and not t.plan_inconsistent
    repaired_or_consistent = not t.plan_inconsistent or t.corrective_replan_succeeds
    intervened = repaired_or_consistent and not t.replan_harm and not t.intervention_harm
    return continued, intervened


TRANSITIONS = {
    "transient_tool_failure": _transition_transient,
    "stale_contradictory_evidence": _transition_stale_evidence,
    "irreversible_action": _transition_irreversible,
    "late_trajectory_failure": _transition_late,
    "replanning_disruption": _transition_replanning,
}


def _classify(continue_passed: bool, intervene_passed: bool) -> str:
    return (
        "recovery" if not continue_passed and intervene_passed else
        "unnecessary_intervention" if continue_passed and intervene_passed else
        "ineffective_intervention" if not continue_passed and not intervene_passed else
        "intervention_harm"
    )


def evaluate_potential_outcomes(
    observation: DecisionObservation,
    evaluator_truth: EvaluatorTruth,
) -> PotentialOutcomes:
    """Generate paired fixture outcomes in the evaluator-only code path."""
    if not isinstance(observation, DecisionObservation):
        raise InvalidSimulatorFixture("evaluator requires the versioned decision observation")
    if not isinstance(evaluator_truth, EvaluatorTruth):
        raise InvalidSimulatorFixture("evaluator truth has an invalid type")
    transition = TRANSITIONS[evaluator_truth.mechanism]
    continued, intervened = transition(evaluator_truth)
    outcome_class = _classify(continued, intervened)
    return PotentialOutcomes(
        continue_passed=continued,
        intervene_passed=intervened,
        outcome_class=outcome_class,
        continue_realized_cost=CONTINUE_COSTS[outcome_class],
        intervene_realized_cost=INTERVENE_COSTS[outcome_class],
    )


def _fixture_truth(mechanism: str, outcome_class: str) -> EvaluatorTruth:
    """Construct unit-test-only states that exercise a requested outcome pair."""
    if mechanism not in MECHANISMS or outcome_class not in OUTCOME_CLASSES:
        raise InvalidSimulatorFixture("unknown mechanism or outcome class")
    c_pass = outcome_class in {"unnecessary_intervention", "intervention_harm"}
    i_pass = outcome_class in {"recovery", "unnecessary_intervention"}
    common = {"mechanism": mechanism}
    if mechanism == "transient_tool_failure":
        return EvaluatorTruth(
            **common,
            base_action_passes=c_pass or (not c_pass and i_pass),
            fault_present=not c_pass,
            repair_succeeds=i_pass,
            intervention_harm=c_pass and not i_pass,
        )
    if mechanism == "stale_contradictory_evidence":
        return EvaluatorTruth(
            **common,
            base_action_passes=c_pass or (not c_pass and i_pass),
            stale_evidence=not c_pass,
            evidence_refresh_succeeds=i_pass,
            intervention_harm=c_pass and not i_pass,
        )
    if mechanism == "irreversible_action":
        return EvaluatorTruth(
            **common,
            pending_action_would_pass=c_pass,
            guarded_replacement_passes=i_pass,
            intervention_harm=c_pass and not i_pass,
        )
    if mechanism == "late_trajectory_failure":
        return EvaluatorTruth(
            **common,
            base_action_passes=c_pass or (not c_pass and i_pass),
            continuation_budget_ok=c_pass,
            intervention_budget_ok=i_pass,
            repair_succeeds=i_pass,
            intervention_harm=c_pass and not i_pass,
        )
    return EvaluatorTruth(
        **common,
        base_action_passes=c_pass or (not c_pass and i_pass),
        plan_inconsistent=not c_pass,
        corrective_replan_succeeds=i_pass,
        replan_harm=c_pass and not i_pass,
    )


def deterministic_outcome_fixture(
    mechanism: str, outcome_class: str,
) -> MechanismFixture:
    """Return a deterministic unit fixture, never a held-out evaluation row."""
    if mechanism not in MECHANISMS or outcome_class not in OUTCOME_CLASSES:
        raise InvalidSimulatorFixture("unknown mechanism or outcome class")
    trajectory_id = "phase9_5_unit_fixture"
    decision_sequence = 0
    evidence = ObservableRiskEvidence(
        anomaly_score=0.4,
        anomaly_alert=False,
        visible_error_count=0,
        visible_warning_count=1,
        source_id="deterministic-unit-fixture",
        source_version="fixture-v1",
        observed_through_sequence=-1,
    )
    observation = build_decision_observation(
        evidence=evidence,
        pending_action_type="reversible_write",
        plan_version="fixture-plan-v1",
        current_step_index=0,
        planned_step_count=2,
        ledger=TrajectoryEventLedger(trajectory_id, complete_from_start=True),
        provenance=ObservationProvenance(
            trajectory_id=trajectory_id,
            # Do not encode evaluator labels or outcome classes in observation
            # metadata: the policy-side snapshot must be category invariant.
            observation_id=f"{mechanism}-fixture",
            decision_sequence=decision_sequence,
            observed_through_sequence=-1,
            plan_source_id="fixture-plan-source-v1",
            action_metadata_source_id="fixture-action-source-v1",
        ),
    )
    return MechanismFixture(observation, _fixture_truth(mechanism, outcome_class))
