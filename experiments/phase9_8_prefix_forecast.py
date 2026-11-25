"""Versioned, offline Phase 9.8-native prefix-forecast comparator.

This simulator-assisted comparator preserves the Phase 9.2 binary forecast
cost arithmetic, not the Phase 9.2 forecast semantics. It deliberately uses
privileged synthetic susceptibility and configuration parameters and is not a
deployment-ready policy. It accepts a narrow immutable context, never an
EvaluatorCase or realized evaluator draws.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping


PROVIDER_VERSION = "phase9_8_native_prefix_forecast_v1"
FEATURE_FIELDS = ("reversibility_proxy", "trajectory_position", "disruption_history")
ROLLOUTS_PER_ACTION = 4
FORECAST_COUNT = ROLLOUTS_PER_ACTION * 2
CONTINUE_COST_MULTIPLIER = 3.0
INTERVENTION_OVERHEAD_COST = 2.0
CONFIGURATION_TABLE_PATH = Path("results/phase9_8_configuration_table.json")
EXPECTED_CONFIGURATION_TABLE_SHA256 = "efc4caf9cf570038a772bb76df57215cc412e3fce0e207566119db6dd83a3963"

# These are the frozen Phase 9.8 configuration-table ranges. They validate
# context only; they do not load or regenerate the table or any outcomes.
PARAMETER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "transient_tool_failure": {
        "repair_probability": (0.55, 0.90), "harm_probability": (0.01, 0.05),
    },
    "stale_contradictory_evidence": {
        "failure_increment": (0.04, 0.12), "repair_probability": (0.45, 0.80),
        "harm_probability": (0.02, 0.08),
    },
    "irreversible_action": {
        "failure_sensitivity": (0.025, 0.10), "repair_probability": (0.30, 0.60),
        "harm_base": (0.02, 0.08), "harm_sensitivity": (0.10, 0.30),
    },
    "late_trajectory_failure": {
        "failure_position_slope": (0.04, 0.12), "repair_intercept": (0.70, 0.90),
        "repair_position_decline": (0.25, 0.55), "harm_intercept": (0.02, 0.05),
        "harm_position_slope": (0.025, 0.075),
    },
    "replanning_disruption": {
        "failure_disruption_slope": (0.06, 0.18), "repair_intercept": (0.65, 0.85),
        "repair_disruption_decline": (0.15, 0.45), "harm_intercept": (0.015, 0.06),
        "harm_disruption_slope": (0.075, 0.225),
    },
}


class ForecastContextError(ValueError):
    """Forecast context is malformed, out of range, or mismatched."""


@lru_cache(maxsize=1)
def _frozen_configuration_index() -> dict[str, tuple[str, tuple[tuple[str, float], ...]]]:
    """Read the hash-verified immutable configuration table once per process."""
    try:
        table = json.loads(CONFIGURATION_TABLE_PATH.read_text(encoding="utf-8"))
        if not isinstance(table, dict):
            raise ForecastContextError("frozen configuration table must be an object")
        recorded = table.get("configuration_table_sha256")
        payload = dict(table)
        payload.pop("configuration_table_sha256", None)
        actual = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ForecastContextError("frozen configuration table is unavailable or invalid") from exc
    if recorded != EXPECTED_CONFIGURATION_TABLE_SHA256 or actual != EXPECTED_CONFIGURATION_TABLE_SHA256:
        raise ForecastContextError("frozen configuration table hash is invalid")
    configurations = table.get("configurations")
    if not isinstance(configurations, list):
        raise ForecastContextError("frozen configuration list is invalid")
    indexed: dict[str, tuple[str, tuple[tuple[str, float], ...]]] = {}
    for row in configurations:
        if not isinstance(row, dict):
            raise ForecastContextError("frozen configuration row is invalid")
        group_id, mechanism, parameters = (
            row.get("configuration_id"), row.get("mechanism"), row.get("parameters"),
        )
        if (not isinstance(group_id, str) or not group_id
                or not isinstance(mechanism, str) or mechanism not in PARAMETER_RANGES
                or not isinstance(parameters, dict) or group_id in indexed):
            raise ForecastContextError("frozen configuration row schema is invalid")
        try:
            frozen_parameters = tuple(sorted((name, float(value)) for name, value in parameters.items()))
        except (TypeError, ValueError) as exc:
            raise ForecastContextError("frozen configuration parameters are invalid") from exc
        indexed[group_id] = (mechanism, frozen_parameters)
    return indexed


@dataclass(frozen=True, slots=True)
class NativePrefixForecastContext:
    """Minimal forecast-time view; evaluator draws and outcomes are excluded."""

    scenario_id: str
    configuration_group_id: str
    root_seed: int
    mechanism: str
    parameters: tuple[tuple[str, float], ...]
    susceptibility: float  # privileged simulator variable; never policy evidence
    reversibility_proxy: float
    trajectory_position: float
    disruption_history: float

    def __post_init__(self) -> None:
        if (not isinstance(self.scenario_id, str) or not self.scenario_id
                or not isinstance(self.configuration_group_id, str)
                or not self.configuration_group_id):
            raise ForecastContextError("scenario and configuration identities must be non-empty")
        if type(self.root_seed) is not int or self.root_seed < 0:
            raise ForecastContextError("root_seed must be a non-negative integer")
        if not isinstance(self.mechanism, str) or self.mechanism not in PARAMETER_RANGES:
            raise ForecastContextError("unknown mechanism")
        if not isinstance(self.parameters, tuple):
            raise ForecastContextError("parameters must be an immutable tuple")
        parameter_map = dict(self.parameters)
        expected = PARAMETER_RANGES[self.mechanism]
        if len(parameter_map) != len(self.parameters) or set(parameter_map) != set(expected):
            raise ForecastContextError("parameter names do not match mechanism schema")
        for name, (low, high) in expected.items():
            value = parameter_map[name]
            _unit_interval(value, f"parameter {name}")
            if not low <= value <= high:
                raise ForecastContextError(f"parameter {name} is outside frozen bounds")
        for field in ("susceptibility", *FEATURE_FIELDS):
            _unit_interval(getattr(self, field), field)
        frozen = _frozen_configuration_index().get(self.configuration_group_id)
        if frozen is None:
            raise ForecastContextError("configuration_group_id is not in the frozen table")
        frozen_mechanism, frozen_parameters = frozen
        if frozen_mechanism != self.mechanism:
            raise ForecastContextError("mechanism does not match frozen configuration group")
        normalized_parameters = tuple(sorted((name, float(value)) for name, value in parameter_map.items()))
        if normalized_parameters != frozen_parameters:
            raise ForecastContextError("parameters do not exactly match frozen configuration group")

    @classmethod
    def create(
        cls, *, scenario_id: str, configuration_group_id: str,
        root_seed: int, mechanism: str,
        parameters: Mapping[str, float], susceptibility: float,
        reversibility_proxy: float, trajectory_position: float,
        disruption_history: float,
    ) -> "NativePrefixForecastContext":
        """Build a validated context from explicit scalar inputs."""
        return cls(
            scenario_id=scenario_id, configuration_group_id=configuration_group_id,
            root_seed=root_seed, mechanism=mechanism,
            parameters=tuple(sorted((str(k), v) for k, v in parameters.items())),
            susceptibility=susceptibility,
            reversibility_proxy=reversibility_proxy,
            trajectory_position=trajectory_position,
            disruption_history=disruption_history,
        )


@dataclass(frozen=True, slots=True)
class ForecastDecision:
    continue_results: tuple[bool, ...]
    intervene_results: tuple[bool, ...]
    continue_pass_count: int
    intervene_pass_count: int
    p_pass_continue: float
    p_pass_intervene: float
    estimated_continue_cost: float
    estimated_intervene_cost: float
    selected_action: str
    forecast_count: int
    forecast_runtime_seconds: float
    forecast_overhead_units: int

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "forecast_provider": PROVIDER_VERSION,
            "baseline_label": "phase9_8_native_simulator_assisted_prefix_forecast",
            "privileged_information": True,
            "forecast_results_continue": list(self.continue_results),
            "forecast_results_intervene": list(self.intervene_results),
            "continue_pass_count": self.continue_pass_count,
            "intervene_pass_count": self.intervene_pass_count,
            "p_pass_continue": self.p_pass_continue,
            "p_pass_intervene": self.p_pass_intervene,
            "estimated_continue_cost": self.estimated_continue_cost,
            "estimated_intervene_cost": self.estimated_intervene_cost,
            "selected_action": self.selected_action,
            "forecast_count": self.forecast_count,
            "forecast_runtime_seconds": self.forecast_runtime_seconds,
            "forecast_overhead_units": self.forecast_overhead_units,
            "overhead_is_separate_from_executed_task_steps_and_realized_task_cost": True,
            "cost_semantics": "Phase 9.2 binary forecast cost structure; Phase 9.8-native forecast semantics",
        }


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastContextError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ForecastContextError(f"{name} must be finite and within [0,1]")
    return result


def _clip_probability(value: float) -> float:
    # Match configured_potential_outcomes: clip once after all raw terms.
    return min(1.0, max(0.0, float(value)))


def outcome_probabilities(context: NativePrefixForecastContext) -> tuple[float, float, float]:
    """Return exactly the Phase 9.8 raw F/R/H equations and clipping order."""
    if not isinstance(context, NativePrefixForecastContext):
        raise TypeError("forecast equations require the narrow forecast context")
    p = dict(context.parameters)
    z = context.susceptibility
    r = context.reversibility_proxy
    x = context.trajectory_position
    d = context.disruption_history
    failure = 0.05 + 0.70 * z
    if context.mechanism == "transient_tool_failure":
        repair, harm = p["repair_probability"], p["harm_probability"]
    elif context.mechanism == "stale_contradictory_evidence":
        failure += p["failure_increment"]
        repair, harm = p["repair_probability"], p["harm_probability"]
    elif context.mechanism == "irreversible_action":
        severity = 1.0 - r
        failure += p["failure_sensitivity"] * severity
        repair = p["repair_probability"]
        harm = p["harm_base"] + p["harm_sensitivity"] * severity
    elif context.mechanism == "late_trajectory_failure":
        failure += p["failure_position_slope"] * x
        repair = p["repair_intercept"] - p["repair_position_decline"] * x
        harm = p["harm_intercept"] + p["harm_position_slope"] * x
    else:  # replanning_disruption
        failure += p["failure_disruption_slope"] * d
        repair = p["repair_intercept"] - p["repair_disruption_decline"] * d
        harm = p["harm_intercept"] + p["harm_disruption_slope"] * d
    values = tuple(_clip_probability(v) for v in (failure, repair, harm))
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise ForecastContextError("equation produced invalid probability")
    return values  # F, R, H


def _draw(context: NativePrefixForecastContext, rollout_index: int,
          stream: str, action: str | None = None) -> float:
    # Canonical JSON + SHA-256 makes each stream independent of execution
    # order. Shared continuation intentionally has no action in its key.
    key = {
        "provider_version": PROVIDER_VERSION,
        "root_seed": context.root_seed,
        "scenario_id": context.scenario_id,
        "rollout_index": rollout_index,
        "stream": stream,
    }
    if action is not None:
        key["action"] = action
    encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    return random.Random(seed).random()


def select_action(continue_cost: float, intervene_cost: float) -> str:
    """Apply the frozen binary-cost rule; ties continue."""
    for value in (continue_cost, intervene_cost):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("estimated costs must be finite numbers")
    return "intervene" if intervene_cost < continue_cost else "continue"


def forecast(context: NativePrefixForecastContext) -> ForecastDecision:
    """Generate four paired forecasts per action, eight forecasts total."""
    if not isinstance(context, NativePrefixForecastContext):
        raise TypeError("forecast requires NativePrefixForecastContext")
    started = time.perf_counter()
    failure, repair, harm = outcome_probabilities(context)
    continue_results: list[bool] = []
    intervene_results: list[bool] = []
    for index in range(ROLLOUTS_PER_ACTION):
        # One continuation draw is shared by both action counterfactuals.
        continued = _draw(context, index, "continuation_shared_v1") >= failure
        repaired = _draw(context, index, "intervention_repair_v1", "intervene") < repair
        unharmed = _draw(context, index, "intervention_harm_v1", "intervene") >= harm
        continue_results.append(continued)
        intervene_results.append((continued or repaired) and unharmed)
    continue_count = sum(continue_results)
    intervene_count = sum(intervene_results)
    p_continue = continue_count / ROLLOUTS_PER_ACTION
    p_intervene = intervene_count / ROLLOUTS_PER_ACTION
    continue_cost = 3.0 * (1.0 - p_continue)
    intervene_cost = 2.0 + 3.0 * (1.0 - p_intervene)
    action = select_action(continue_cost, intervene_cost)
    return ForecastDecision(
        tuple(continue_results), tuple(intervene_results), continue_count, intervene_count,
        p_continue, p_intervene, continue_cost, intervene_cost, action,
        FORECAST_COUNT, time.perf_counter() - started, FORECAST_COUNT,
    )
