"""Immutable append-only event history for Phase 9.5 decision snapshots.

This module contains only event facts observable by the trajectory runner.
It deliberately has no access to simulator truth or evaluator outcomes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


EVENT_SCHEMA_VERSION = "phase9-5.1-trajectory-events-v1"
COUNTED_EVENT = "intervention_applied"
ALLOWED_EVENT_TYPES = frozenset({
    "intervention_applied", "replan_started", "replan_completed", "replan_failed",
})


class InvalidEventHistory(ValueError):
    """Raised when a decision-time event prefix is incomplete or inconsistent."""


@dataclass(frozen=True)
class TrajectoryEvent:
    event_id: str
    trajectory_id: str
    sequence: int
    event_type: str
    intervention_id: str
    plan_version: str

    def __post_init__(self) -> None:
        if not self.event_id or not self.trajectory_id or not self.intervention_id or not self.plan_version:
            raise InvalidEventHistory("event identifiers and plan version are required")
        if type(self.sequence) is not int or self.sequence < 0:
            raise InvalidEventHistory("event sequence must be a non-negative integer")
        if self.event_type not in ALLOWED_EVENT_TYPES:
            raise InvalidEventHistory("unsupported trajectory event type")


@dataclass(frozen=True)
class TrajectoryEventLedger:
    trajectory_id: str
    complete_from_start: bool
    events: tuple[TrajectoryEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise InvalidEventHistory("trajectory identifier is required")
        if type(self.complete_from_start) is not bool:
            raise InvalidEventHistory("history completeness must be boolean")
        if not isinstance(self.events, tuple):
            raise InvalidEventHistory("ledger events must be an immutable tuple")
        if any(not isinstance(event, TrajectoryEvent) for event in self.events):
            raise InvalidEventHistory("ledger contains a malformed event")

    def append(self, event: TrajectoryEvent) -> "TrajectoryEventLedger":
        """Return a new ledger with one appended event; the original is unchanged."""
        if event.trajectory_id != self.trajectory_id:
            raise InvalidEventHistory("event belongs to a different trajectory")
        if event.sequence != len(self.events):
            raise InvalidEventHistory("append sequence must be contiguous")
        return TrajectoryEventLedger(
            trajectory_id=self.trajectory_id,
            complete_from_start=self.complete_from_start,
            events=self.events + (event,),
        )


@dataclass(frozen=True)
class DisruptionHistory:
    trajectory_id: str
    decision_sequence: int
    applied_intervention_count: int
    value: float
    event_ids: tuple[str, ...]
    prefix_sha256: str


def _validate_prefix(events: tuple[TrajectoryEvent, ...], trajectory_id: str,
                     decision_sequence: int) -> None:
    if len(events) != decision_sequence:
        raise InvalidEventHistory("event prefix has missing or gapped sequence numbers")
    seen_event_ids: set[str] = set()
    applied_ids: set[str] = set()
    replans: dict[str, str] = {}
    for expected_sequence, event in enumerate(events):
        if event.trajectory_id != trajectory_id or event.sequence != expected_sequence:
            raise InvalidEventHistory("event prefix is out of order or crosses trajectories")
        if event.event_id in seen_event_ids:
            raise InvalidEventHistory("duplicate event identifier")
        seen_event_ids.add(event.event_id)
        intervention_id = event.intervention_id
        if event.event_type == COUNTED_EVENT:
            if intervention_id in applied_ids:
                raise InvalidEventHistory("duplicate intervention application")
            applied_ids.add(intervention_id)
            if intervention_id in replans:
                raise InvalidEventHistory("intervention application follows its replan")
        elif event.event_type == "replan_started":
            if intervention_id not in applied_ids or intervention_id in replans:
                raise InvalidEventHistory("replan start has no unique prior intervention")
            replans[intervention_id] = "started"
        else:
            if replans.get(intervention_id) != "started":
                raise InvalidEventHistory("replan terminal event has no unmatched start")
            replans[intervention_id] = event.event_type
    if "started" in replans.values():
        raise InvalidEventHistory("replan lifecycle is incomplete at the decision")


def reduce_disruption_history(ledger: TrajectoryEventLedger,
                              decision_sequence: int) -> DisruptionHistory:
    """Reduce only events strictly before the decision into the approved feature.

    The event at ``decision_sequence`` and all later events are ignored. A
    complete empty prefix is a valid zero; a partial prefix is an error.
    """
    if type(decision_sequence) is not int or decision_sequence < 0:
        raise InvalidEventHistory("decision sequence must be a non-negative integer")
    if not ledger.complete_from_start:
        raise InvalidEventHistory("event history is not complete from trajectory start")
    prefix = tuple(event for event in ledger.events if event.sequence < decision_sequence)
    _validate_prefix(prefix, ledger.trajectory_id, decision_sequence)
    counted = tuple(event for event in prefix if event.event_type == COUNTED_EVENT)
    payload = [
        [event.event_id, event.trajectory_id, event.sequence, event.event_type,
         event.intervention_id, event.plan_version]
        for event in prefix
    ]
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    count = len(counted)
    return DisruptionHistory(
        trajectory_id=ledger.trajectory_id,
        decision_sequence=decision_sequence,
        applied_intervention_count=count,
        value=min(count / 3.0, 1.0),
        event_ids=tuple(event.event_id for event in counted),
        prefix_sha256=digest,
    )
