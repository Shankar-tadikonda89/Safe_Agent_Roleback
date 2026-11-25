"""Versioned Phase 9.7 synthetic observation and evaluator simulator.

This is a design-assumption simulator, not an empirical sensor model. The
visible score formula is inherited from Phase 9.5 and remains explicitly
uncalibrated. All numeric parameters below are frozen illustrative assumptions
for a future protocol; they were not selected using held-out outcomes.

Common-cause model (illustrative): draw hidden susceptibility z ~ U(0, 1).
The pre-decision synthetic monitor observes noisy, probabilistic measurements
of z: anomaly=clip(0.15 + 0.70*z + Normal(0, 0.15)); alert probability is
0.05 + 0.85*z; each of two visible error indicators has probability
0.05 + 0.55*z; each of two warning indicators has probability 0.10 + 0.45*z.
The same z affects evaluator continuation failure risk, but none of z, fault
flags, outcomes, labels, or evaluator draws is present in PolicyScenario.
Sensor and outcome draws use separate SHA-256-derived random streams.

Mechanism parameters are illustrative and fixed: base continuation failure
risk is 0.05 + 0.70*z. The five mechanisms add the documented modifiers in
``_outcome_parameters``. A shared continuation result is used for paired
potential outcomes; repair and intervention-harm draws are independently
domain-separated. These assumptions are not evidence of real-world or live
LLM-agent effectiveness. Do not tune them after held-out outcomes exist.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from experiments.phase9_5_event_history import (
    InvalidEventHistory,
    TrajectoryEvent,
    TrajectoryEventLedger,
    reduce_disruption_history,
)
from experiments.phase9_5_observations import (
    FAILURE_HORIZON,
    FAILURE_TARGET,
    REVERSIBILITY_VALUES,
    RISK_SIGNAL_KIND,
    InvalidObservation,
    ObservableRiskEvidence,
    ObservationProvenance,
    DecisionObservation,
    build_decision_observation,
    extract_features,
)


SIMULATOR_VERSION = "phase9-7-synthetic-prefix-monitor-v1"
RISK_SOURCE_ID = "synthetic-prefix-monitor-v1"
RISK_SOURCE_VERSION = "synthetic-prefix-monitor-v1"
STREAM_NAMES = frozenset({
    "latent_common_cause", "sensor_noise", "sensor_alert", "visible_errors",
    "visible_warnings", "plan_action", "history", "continuation_execution",
    "repair_success", "intervention_harm",
})
MECHANISMS = (
    "transient_tool_failure",
    "stale_contradictory_evidence",
    "irreversible_action",
    "late_trajectory_failure",
    "replanning_disruption",
)
OUTCOME_CLASSES = (
    "recovery", "unnecessary_intervention", "ineffective_intervention", "intervention_harm",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(OUTCOME_CLASSES)}

# Phase 9 frozen values, copied without changes.
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

_CONTINUE_BASE_FAILURE = 0.05
_CONTINUE_SUSCEPTIBILITY_SLOPE = 0.70
_SENSOR_ANOMALY_INTERCEPT = 0.15
_SENSOR_ANOMALY_SLOPE = 0.70
_SENSOR_NOISE_SD = 0.15
_ALERT_FALSE_POSITIVE = 0.05
_ALERT_SENSITIVITY = 0.85
_ERROR_BASE_RATE = 0.05
_ERROR_SENSITIVITY = 0.55
_WARNING_BASE_RATE = 0.10
_WARNING_SENSITIVITY = 0.45


class InvalidScenario(ValueError):
    """Raised when a public scenario or evaluator fixture violates its schema."""


def _unit(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidScenario(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise InvalidScenario(f"{label} must be finite and in [0, 1]")
    return result


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _stream_seed(root_seed: int, scenario_id: str, stream: str) -> int:
    if type(root_seed) is not int or root_seed < 0:
        raise InvalidScenario("root_seed must be a non-negative integer")
    if not scenario_id or stream not in STREAM_NAMES:
        raise InvalidScenario("scenario id or random stream is invalid")
    payload = json.dumps(
        [SIMULATOR_VERSION, root_seed, scenario_id, stream],
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng(root_seed: int, scenario_id: str, stream: str) -> random.Random:
    return random.Random(_stream_seed(root_seed, scenario_id, stream))


@dataclass(frozen=True)
class TimedEvent:
    event: TrajectoryEvent
    logical_tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.event, TrajectoryEvent):
            raise InvalidEventHistory("timed event must wrap a trajectory event")
        if type(self.logical_tick) is not int or self.logical_tick < 0:
            raise InvalidEventHistory("logical event tick must be a non-negative integer")


@dataclass(frozen=True)
class TimedTrajectoryLedger:
    trajectory_id: str
    complete_from_start: bool = True
    events: tuple[TimedEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.trajectory_id or type(self.complete_from_start) is not bool:
            raise InvalidEventHistory("trajectory id and completeness declaration are required")
        if not isinstance(self.events, tuple) or any(not isinstance(item, TimedEvent) for item in self.events):
            raise InvalidEventHistory("timed ledger must contain an immutable tuple of events")
        prior_tick = -1
        for sequence, item in enumerate(self.events):
            if item.event.trajectory_id != self.trajectory_id or item.event.sequence != sequence:
                raise InvalidEventHistory("timed ledger has a sequence gap or trajectory mismatch")
            if item.logical_tick <= prior_tick:
                raise InvalidEventHistory("event logical ticks must be strictly increasing")
            prior_tick = item.logical_tick

    def append(self, item: TimedEvent) -> "TimedTrajectoryLedger":
        if not isinstance(item, TimedEvent):
            raise InvalidEventHistory("append requires a timed event")
        if item.event.trajectory_id != self.trajectory_id or item.event.sequence != len(self.events):
            raise InvalidEventHistory("appended event has wrong trajectory or sequence")
        if self.events and item.logical_tick <= self.events[-1].logical_tick:
            raise InvalidEventHistory("appended event must have a later logical tick")
        return TimedTrajectoryLedger(
            trajectory_id=self.trajectory_id,
            complete_from_start=self.complete_from_start,
            events=self.events + (item,),
        )

    def untimed_prefix(self) -> TrajectoryEventLedger:
        return TrajectoryEventLedger(
            trajectory_id=self.trajectory_id,
            complete_from_start=self.complete_from_start,
            events=tuple(item.event for item in self.events),
        )


def _prefix_digest(events: tuple[TimedEvent, ...]) -> str:
    payload = [
        [item.event.event_id, item.event.trajectory_id, item.event.sequence,
         item.event.event_type, item.event.intervention_id, item.event.plan_version,
         item.logical_tick]
        for item in events
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class DecisionSnapshot:
    """Runner-captured immutable view through exactly the event before decision."""
    observation: DecisionObservation
    event_prefix: tuple[TimedEvent, ...]
    evidence_timestamp_tick: int
    decision_timestamp_tick: int
    observed_through_sequence: int
    prefix_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, DecisionObservation):
            raise InvalidObservation("snapshot requires a typed decision observation")
        if not isinstance(self.event_prefix, tuple) or any(not isinstance(e, TimedEvent) for e in self.event_prefix):
            raise InvalidEventHistory("snapshot prefix must be an immutable timed-event tuple")
        decision_sequence = self.observation.provenance.decision_sequence
        expected_observed = decision_sequence - 1
        if self.observed_through_sequence != expected_observed:
            raise InvalidObservation("snapshot is stale, future, or inconsistent with the decision sequence")
        if self.observation.provenance.observed_through_sequence != expected_observed:
            raise InvalidObservation("observation provenance does not match the current prefix")
        if len(self.event_prefix) != decision_sequence:
            raise InvalidEventHistory("snapshot must contain the complete prefix and no future events")
        if type(self.evidence_timestamp_tick) is not int or type(self.decision_timestamp_tick) is not int:
            raise InvalidObservation("snapshot timestamps must be logical integer ticks")
        if self.evidence_timestamp_tick < 0 or self.evidence_timestamp_tick != self.decision_timestamp_tick:
            raise InvalidObservation("evidence must be captured at the decision snapshot tick")
        prior_tick = -1
        for sequence, item in enumerate(self.event_prefix):
            if item.event.sequence != sequence:
                raise InvalidEventHistory("snapshot prefix has a sequence gap or future event")
            if item.logical_tick <= prior_tick or item.logical_tick >= self.evidence_timestamp_tick:
                raise InvalidEventHistory("snapshot event timestamps are stale, unordered, or future")
            prior_tick = item.logical_tick
        signal = self.observation.risk_signal
        if signal.source_id != RISK_SOURCE_ID or signal.source_version != RISK_SOURCE_VERSION:
            raise InvalidObservation("snapshot risk signal is not from the frozen synthetic monitor")
        if signal.kind != RISK_SIGNAL_KIND or signal.target != FAILURE_TARGET or signal.horizon != FAILURE_HORIZON:
            raise InvalidObservation("snapshot risk signal semantics do not match the frozen contract")
        if self.prefix_sha256 != _prefix_digest(self.event_prefix):
            raise InvalidEventHistory("snapshot prefix digest mismatch")
        ledger = TrajectoryEventLedger(
            trajectory_id=self.observation.provenance.trajectory_id,
            complete_from_start=True,
            events=tuple(item.event for item in self.event_prefix),
        )
        reduced = reduce_disruption_history(ledger, decision_sequence)
        if reduced != self.observation.disruption:
            raise InvalidEventHistory("snapshot disruption feature does not match its event prefix")


def capture_decision_snapshot(
    *, evidence: ObservableRiskEvidence, pending_action_type: str,
    plan_version: str, current_step_index: int, planned_step_count: int,
    ledger: TimedTrajectoryLedger, evidence_timestamp_tick: int,
    decision_timestamp_tick: int, observed_through_sequence: int | None = None,
    decision_sequence: int | None = None, observation_id: str = "snapshot",
) -> DecisionSnapshot:
    """Validate a runner-owned, current snapshot before feature extraction."""
    if not isinstance(ledger, TimedTrajectoryLedger) or not ledger.complete_from_start:
        raise InvalidEventHistory("a complete runner-owned ledger is required")
    d = len(ledger.events) if decision_sequence is None else decision_sequence
    observed = d - 1 if observed_through_sequence is None else observed_through_sequence
    if type(d) is not int or d < 0 or d != len(ledger.events):
        raise InvalidObservation("decision sequence must equal the captured ledger prefix length")
    if type(observed) is not int or observed != d - 1:
        raise InvalidObservation("stale or future observation sequence is rejected")
    if type(evidence_timestamp_tick) is not int or type(decision_timestamp_tick) is not int:
        raise InvalidObservation("timestamps must be integer logical ticks")
    if evidence_timestamp_tick != decision_timestamp_tick:
        raise InvalidObservation("evidence timestamp must equal the decision snapshot timestamp")
    if ledger.events and ledger.events[-1].logical_tick >= evidence_timestamp_tick:
        raise InvalidEventHistory("snapshot timestamp does not follow its event prefix")
    if evidence.source_id != RISK_SOURCE_ID or evidence.source_version != RISK_SOURCE_VERSION:
        raise InvalidObservation("risk evidence source is not the frozen synthetic monitor")
    if evidence.observed_through_sequence != observed:
        raise InvalidObservation("evidence and snapshot sequence provenance disagree")

    untimed = ledger.untimed_prefix()
    observation = build_decision_observation(
        evidence=evidence,
        pending_action_type=pending_action_type,
        plan_version=plan_version,
        current_step_index=current_step_index,
        planned_step_count=planned_step_count,
        ledger=untimed,
        provenance=ObservationProvenance(
            trajectory_id=ledger.trajectory_id,
            observation_id=observation_id,
            decision_sequence=d,
            observed_through_sequence=observed,
            plan_source_id="runner-visible-plan-v1",
            action_metadata_source_id="runner-pending-action-v1",
        ),
    )
    return DecisionSnapshot(
        observation=observation,
        event_prefix=ledger.events,
        evidence_timestamp_tick=evidence_timestamp_tick,
        decision_timestamp_tick=decision_timestamp_tick,
        observed_through_sequence=observed,
        prefix_sha256=_prefix_digest(ledger.events),
    )


def append_applied_intervention(
    ledger: TimedTrajectoryLedger, *, intervention_id: str, plan_version: str,
    first_tick: int, actually_applied: bool, replan_outcome: str = "completed",
) -> TimedTrajectoryLedger:
    """Append applied/start/terminal events only after confirmed application."""
    if actually_applied is not True:
        raise InvalidEventHistory("intervention_applied may be recorded only after actual application")
    if replan_outcome not in {"completed", "failed"}:
        raise InvalidEventHistory("replan outcome must be completed or failed")
    if type(first_tick) is not int or first_tick < 0:
        raise InvalidEventHistory("first event tick must be non-negative")
    current = ledger
    event_types = ("intervention_applied", "replan_started", f"replan_{replan_outcome}")
    for offset, event_type in enumerate(event_types):
        sequence = len(current.events)
        event = TrajectoryEvent(
            event_id=f"{intervention_id}:{event_type}",
            trajectory_id=current.trajectory_id,
            sequence=sequence,
            event_type=event_type,
            intervention_id=intervention_id,
            plan_version=plan_version,
        )
        current = current.append(TimedEvent(event, first_tick + offset))
    # Validate lifecycle with the same locked prefix reducer used by Phase 9.5.
    reduce_disruption_history(current.untimed_prefix(), len(current.events))
    return current


@dataclass(frozen=True)
class PolicyScenario:
    """Policy-side data only; evaluator truth is deliberately not a field."""
    scenario_id: str
    initial_state_hash: str
    snapshot: DecisionSnapshot

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.initial_state_hash:
            raise InvalidScenario("scenario id and initial-state digest are required")
        if not isinstance(self.snapshot, DecisionSnapshot):
            raise InvalidScenario("policy scenario requires a valid decision snapshot")


@dataclass(frozen=True)
class EvaluatorCase:
    """Hidden evaluator data; never included in PolicyScenario or policy input."""
    scenario_id: str
    mechanism: str
    susceptibility: float
    continuation_uniform: float
    repair_uniform: float
    harm_uniform: float

    def __post_init__(self) -> None:
        if not self.scenario_id or self.mechanism not in MECHANISMS:
            raise InvalidScenario("evaluator scenario or mechanism is invalid")
        for field in ("susceptibility", "continuation_uniform", "repair_uniform", "harm_uniform"):
            _unit(getattr(self, field), field)


@dataclass(frozen=True)
class GeneratedScenario:
    """Builder result split into independently passed public/evaluator views."""
    policy: PolicyScenario
    evaluator: EvaluatorCase

    def __post_init__(self) -> None:
        if self.policy.scenario_id != self.evaluator.scenario_id:
            raise InvalidScenario("public and evaluator records are not scenario-matched")


@dataclass(frozen=True)
class PotentialOutcomes:
    continue_passed: bool
    intervene_passed: bool
    outcome_class: str
    continue_realized_cost: float
    intervene_realized_cost: float
    evaluator_only: bool = True


def _outcome_parameters(mechanism: str, susceptibility: float,
                        reversibility: float, position: float,
                        disruption: float) -> tuple[float, float, float]:
    """Return illustrative (continuation failure, repair, harm) probabilities."""
    failure = _CONTINUE_BASE_FAILURE + _CONTINUE_SUSCEPTIBILITY_SLOPE * susceptibility
    repair = 0.75
    harm = 0.02
    if mechanism == "stale_contradictory_evidence":
        failure += 0.08
        repair, harm = 0.65, 0.04
    elif mechanism == "irreversible_action":
        severity = 1.0 - reversibility
        failure += 0.05 * severity
        repair, harm = 0.45, 0.04 + 0.20 * severity
    elif mechanism == "late_trajectory_failure":
        failure += 0.08 * position
        repair = 0.80 - 0.45 * position
        harm = 0.03 + 0.05 * position
    elif mechanism == "replanning_disruption":
        failure += 0.12 * disruption
        repair = 0.75 - 0.30 * disruption
        harm = 0.03 + 0.15 * disruption
    return _clamp(failure), _clamp(repair), _clamp(harm)


def classify_outcome(continue_passed: bool, intervene_passed: bool) -> str:
    if type(continue_passed) is not bool or type(intervene_passed) is not bool:
        raise InvalidScenario("potential verifier outcomes must be booleans")
    if not continue_passed and intervene_passed:
        return "recovery"
    if continue_passed and intervene_passed:
        return "unnecessary_intervention"
    if not continue_passed and not intervene_passed:
        return "ineffective_intervention"
    return "intervention_harm"


def evaluate_potential_outcomes(
    policy: PolicyScenario, evaluator: EvaluatorCase,
) -> PotentialOutcomes:
    """Evaluator-only paired outcome calculation; never called by policy inference."""
    if not isinstance(policy, PolicyScenario) or not isinstance(evaluator, EvaluatorCase):
        raise InvalidScenario("evaluator requires distinct public and evaluator records")
    if policy.scenario_id != evaluator.scenario_id:
        raise InvalidScenario("policy and evaluator scenarios do not match")
    features = extract_features(policy.snapshot.observation)
    failure_probability, repair_probability, harm_probability = _outcome_parameters(
        evaluator.mechanism, evaluator.susceptibility,
        features.reversibility_proxy, features.trajectory_position,
        features.disruption_history,
    )
    continued = evaluator.continuation_uniform >= failure_probability
    pre_harm_intervention_pass = (
        continued or evaluator.repair_uniform < repair_probability
    )
    intervened = pre_harm_intervention_pass and evaluator.harm_uniform >= harm_probability
    outcome = classify_outcome(continued, intervened)
    return PotentialOutcomes(
        continue_passed=continued,
        intervene_passed=intervened,
        outcome_class=outcome,
        continue_realized_cost=CONTINUE_COSTS[outcome],
        intervene_realized_cost=INTERVENE_COSTS[outcome],
    )


def build_scenario(
    *, root_seed: int, scenario_seed: int, mechanism: str,
    trajectory_id: str | None = None,
) -> GeneratedScenario:
    """Build one synthetic public prefix and its separate evaluator record."""
    if type(scenario_seed) is not int or scenario_seed < 0:
        raise InvalidScenario("scenario_seed must be a non-negative integer")
    if mechanism not in MECHANISMS:
        raise InvalidScenario("unknown mechanism")
    scenario_id = hashlib.sha256(json.dumps(
        [SIMULATOR_VERSION, root_seed, scenario_seed, mechanism], separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:20]
    trajectory = trajectory_id or f"trajectory-{scenario_id}"
    latent_rng = _rng(root_seed, scenario_id, "latent_common_cause")
    susceptibility = latent_rng.random()

    noise = _rng(root_seed, scenario_id, "sensor_noise").gauss(0.0, _SENSOR_NOISE_SD)
    anomaly = _clamp(_SENSOR_ANOMALY_INTERCEPT + _SENSOR_ANOMALY_SLOPE * susceptibility + noise)
    alert_probability = _ALERT_FALSE_POSITIVE + _ALERT_SENSITIVITY * susceptibility
    alert = _rng(root_seed, scenario_id, "sensor_alert").random() < alert_probability
    error_rng = _rng(root_seed, scenario_id, "visible_errors")
    visible_errors = sum(
        error_rng.random() < _ERROR_BASE_RATE + _ERROR_SENSITIVITY * susceptibility
        for _ in range(2)
    )
    warning_rng = _rng(root_seed, scenario_id, "visible_warnings")
    visible_warnings = sum(
        warning_rng.random() < _WARNING_BASE_RATE + _WARNING_SENSITIVITY * susceptibility
        for _ in range(2)
    )

    plan_rng = _rng(root_seed, scenario_id, "plan_action")
    planned_steps = plan_rng.randint(2, 6)
    current_step = plan_rng.randrange(planned_steps)
    action_draw = plan_rng.random()
    action_type = (
        "read_only" if action_draw < 0.40 else
        "reversible_write" if action_draw < 0.80 else
        "irreversible_write"
    )

    # These are actual prior synthetic intervention/replan events, not a hidden
    # count injected directly as a feature.
    history_count = _rng(root_seed, scenario_id, "history").randint(0, 3)
    ledger = TimedTrajectoryLedger(trajectory_id=trajectory)
    for index in range(history_count):
        intervention_id = f"prior-{index}"
        ledger = append_applied_intervention(
            ledger,
            intervention_id=intervention_id,
            plan_version=f"prior-plan-{index}",
            first_tick=index * 3,
            actually_applied=True,
            replan_outcome="completed",
        )

    decision_tick = max(0, 3 * history_count)
    decision_sequence = len(ledger.events)
    observed_through = decision_sequence - 1
    evidence = ObservableRiskEvidence(
        anomaly_score=anomaly,
        anomaly_alert=alert,
        visible_error_count=visible_errors,
        visible_warning_count=visible_warnings,
        source_id=RISK_SOURCE_ID,
        source_version=RISK_SOURCE_VERSION,
        observed_through_sequence=observed_through,
    )
    snapshot = capture_decision_snapshot(
        evidence=evidence,
        pending_action_type=action_type,
        plan_version="visible-plan-v1",
        current_step_index=current_step,
        planned_step_count=planned_steps,
        ledger=ledger,
        evidence_timestamp_tick=decision_tick,
        decision_timestamp_tick=decision_tick,
        observed_through_sequence=observed_through,
        decision_sequence=decision_sequence,
        observation_id=f"obs-{scenario_id}",
    )
    initial_state_hash = hashlib.sha256(b"phase9-7-empty-initial-state-v1").hexdigest()
    policy = PolicyScenario(scenario_id, initial_state_hash, snapshot)
    evaluator = EvaluatorCase(
        scenario_id=scenario_id,
        mechanism=mechanism,
        susceptibility=susceptibility,
        continuation_uniform=_rng(root_seed, scenario_id, "continuation_execution").random(),
        repair_uniform=_rng(root_seed, scenario_id, "repair_success").random(),
        harm_uniform=_rng(root_seed, scenario_id, "intervention_harm").random(),
    )
    return GeneratedScenario(policy=policy, evaluator=evaluator)


def reserve_new_output_path(path: str | Path, *, kind: str) -> Path:
    """Validate a non-existing Phase 9.7 output path without creating it."""
    roots = {
        "results": Path("results/phase9_7"),
        "models": Path("models/phase9_7"),
    }
    if kind not in roots:
        raise ValueError("output kind must be results or models")
    target = Path(path).resolve()
    allowed_root = roots[kind].resolve()
    try:
        target.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"new {kind} output must stay under {allowed_root}") from exc
    if target.exists():
        raise FileExistsError(f"refusing to reuse existing Phase 9.7 output path: {target}")
    return target

