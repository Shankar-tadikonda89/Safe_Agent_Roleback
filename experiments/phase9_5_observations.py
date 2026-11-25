"""Typed Phase 9.5 decision observations and provenance.

The legacy feature key ``failure_prob`` is retained for schema compatibility,
but this module emits only an explicitly uncalibrated synthetic risk score.
It must not be reported or interpreted as a calibrated probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from experiments.phase9_5_event_history import (
    DisruptionHistory,
    TrajectoryEventLedger,
    reduce_disruption_history,
)


OBSERVATION_SCHEMA_VERSION = "phase9-5.1-observation-v1"
RISK_SIGNAL_KIND = "uncalibrated_synthetic_risk_score"
RISK_SIGNAL_SOURCE = "phase9_5_visible_fixture_evidence"
RISK_SIGNAL_SOURCE_VERSION = "visible-evidence-score-v1"
FAILURE_TARGET = "next_verifier_checkpoint_rejects_without_intervention"
FAILURE_HORIZON = "next_verifier_checkpoint_or_final_verification"
REVERSIBILITY_MAPPING_VERSION = "action-reversibility-v1"
FEATURE_ORDER = (
    "failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history",
)
REVERSIBILITY_VALUES = {
    "read_only": 0.95,
    "reversible_write": 0.65,
    "irreversible_write": 0.20,
}


class InvalidObservation(ValueError):
    """Raised instead of coercing, defaulting, or silently inferring missing data."""


def _finite_unit(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidObservation(f"{label} must be a numeric value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise InvalidObservation(f"{label} must be finite and within [0, 1]")
    return result


@dataclass(frozen=True)
class ObservableRiskEvidence:
    """Fixture-visible evidence, not hidden simulator state or an oracle result."""
    anomaly_score: float
    anomaly_alert: bool
    visible_error_count: int
    visible_warning_count: int
    source_id: str
    source_version: str
    observed_through_sequence: int

    def __post_init__(self) -> None:
        _finite_unit(self.anomaly_score, "anomaly_score")
        if type(self.anomaly_alert) is not bool:
            raise InvalidObservation("anomaly_alert must be boolean")
        for name, value in (("visible_error_count", self.visible_error_count),
                            ("visible_warning_count", self.visible_warning_count),
                            ("observed_through_sequence", self.observed_through_sequence)):
            lower_bound = -1 if name == "observed_through_sequence" else 0
            if type(value) is not int or value < lower_bound:
                raise InvalidObservation(f"{name} must be a non-negative integer")
        if not self.source_id or not self.source_version:
            raise InvalidObservation("risk evidence source provenance is required")


@dataclass(frozen=True)
class ObservationProvenance:
    trajectory_id: str
    observation_id: str
    decision_sequence: int
    observed_through_sequence: int
    plan_source_id: str
    action_metadata_source_id: str

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.observation_id:
            raise InvalidObservation("trajectory and observation identifiers are required")
        if not self.plan_source_id or not self.action_metadata_source_id:
            raise InvalidObservation("plan and action metadata provenance are required")
        if type(self.decision_sequence) is not int or self.decision_sequence < 0:
            raise InvalidObservation("decision_sequence must be a non-negative integer")
        if type(self.observed_through_sequence) is not int:
            raise InvalidObservation("observed_through_sequence must be an integer")
        if self.observed_through_sequence < -1 or self.observed_through_sequence >= self.decision_sequence:
            raise InvalidObservation("observation provenance includes the decision or a future event")


@dataclass(frozen=True)
class SyntheticRiskSignal:
    value: float
    kind: str
    source_id: str
    source_version: str
    target: str
    horizon: str
    observed_through_sequence: int

    def __post_init__(self) -> None:
        _finite_unit(self.value, "failure_prob synthetic score")
        if self.kind != RISK_SIGNAL_KIND:
            raise InvalidObservation("risk signal must be explicitly marked uncalibrated")
        if self.target != FAILURE_TARGET or self.horizon != FAILURE_HORIZON:
            raise InvalidObservation("risk signal target or horizon does not match the contract")
        if not self.source_id or not self.source_version:
            raise InvalidObservation("risk signal source provenance is required")
        if type(self.observed_through_sequence) is not int or self.observed_through_sequence < -1:
            raise InvalidObservation("risk signal observation sequence is invalid")


@dataclass(frozen=True)
class DecisionObservation:
    """Immutable, prefix-only input from which the locked features are derived."""
    provenance: ObservationProvenance
    risk_signal: SyntheticRiskSignal
    pending_action_type: str
    plan_version: str
    current_step_index: int
    planned_step_count: int
    disruption: DisruptionHistory
    reversibility_mapping_version: str = REVERSIBILITY_MAPPING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.pending_action_type, str) or self.pending_action_type not in REVERSIBILITY_VALUES:
            raise InvalidObservation("pending action type is missing or unknown")
        if self.reversibility_mapping_version != REVERSIBILITY_MAPPING_VERSION:
            raise InvalidObservation("unsupported reversibility mapping version")
        if not self.plan_version:
            raise InvalidObservation("visible plan version is required")
        if type(self.planned_step_count) is not int or self.planned_step_count <= 0:
            raise InvalidObservation("empty or invalid plan is not a decision observation")
        if type(self.current_step_index) is not int or not 0 <= self.current_step_index < self.planned_step_count:
            raise InvalidObservation("current step index is outside the visible plan")
        if self.provenance.trajectory_id != self.disruption.trajectory_id:
            raise InvalidObservation("history belongs to a different trajectory")
        if self.provenance.decision_sequence != self.disruption.decision_sequence:
            raise InvalidObservation("history prefix does not end at this decision")
        if self.risk_signal.observed_through_sequence > self.provenance.observed_through_sequence:
            raise InvalidObservation("risk signal provenance extends beyond the observation snapshot")


@dataclass(frozen=True)
class FeatureVector:
    """Immutable four-feature vector; ``failure_prob`` is an uncalibrated score here."""
    failure_prob: float
    reversibility_proxy: float
    trajectory_position: float
    disruption_history: float
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    failure_signal_kind: str = RISK_SIGNAL_KIND

    def __post_init__(self) -> None:
        for name in FEATURE_ORDER:
            _finite_unit(getattr(self, name), name)
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise InvalidObservation("unsupported feature schema version")
        if self.failure_signal_kind != RISK_SIGNAL_KIND:
            raise InvalidObservation("feature vector has unapproved risk semantics")

    def as_ordered_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.failure_prob,
            self.reversibility_proxy,
            self.trajectory_position,
            self.disruption_history,
        )

    def as_ordered_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_ORDER, self.as_ordered_tuple(), strict=True))


def score_visible_evidence(evidence: ObservableRiskEvidence) -> SyntheticRiskSignal:
    """Create a transparent synthetic score; the fixed weights are not calibrated."""
    error_component = min(evidence.visible_error_count, 2) / 2.0
    warning_component = min(evidence.visible_warning_count, 2) / 2.0
    score = (
        0.25 * evidence.anomaly_score
        + 0.25 * float(evidence.anomaly_alert)
        + 0.25 * error_component
        + 0.25 * warning_component
    )
    return SyntheticRiskSignal(
        value=score,
        kind=RISK_SIGNAL_KIND,
        source_id=evidence.source_id,
        source_version=evidence.source_version,
        target=FAILURE_TARGET,
        horizon=FAILURE_HORIZON,
        observed_through_sequence=evidence.observed_through_sequence,
    )


def build_decision_observation(
    *,
    evidence: ObservableRiskEvidence,
    pending_action_type: str,
    plan_version: str,
    current_step_index: int,
    planned_step_count: int,
    ledger: TrajectoryEventLedger,
    provenance: ObservationProvenance,
) -> DecisionObservation:
    """Build a decision snapshot from visible evidence and a validated ledger prefix."""
    if evidence.observed_through_sequence != provenance.observed_through_sequence:
        raise InvalidObservation("risk evidence and snapshot provenance disagree")
    if ledger.trajectory_id != provenance.trajectory_id:
        raise InvalidObservation("event ledger belongs to a different trajectory")
    disruption = reduce_disruption_history(ledger, provenance.decision_sequence)
    risk = score_visible_evidence(evidence)
    return DecisionObservation(
        provenance=provenance,
        risk_signal=risk,
        pending_action_type=pending_action_type,
        plan_version=plan_version,
        current_step_index=current_step_index,
        planned_step_count=planned_step_count,
        disruption=disruption,
    )


def extract_features(observation: DecisionObservation) -> FeatureVector:
    """Extract only the four approved fields; never accepts evaluator state."""
    if not isinstance(observation, DecisionObservation):
        raise InvalidObservation("feature extraction requires a DecisionObservation")
    position = (
        0.0 if observation.planned_step_count == 1
        else observation.current_step_index / (observation.planned_step_count - 1)
    )
    return FeatureVector(
        failure_prob=observation.risk_signal.value,
        reversibility_proxy=REVERSIBILITY_VALUES[observation.pending_action_type],
        trajectory_position=position,
        disruption_history=observation.disruption.value,
    )
