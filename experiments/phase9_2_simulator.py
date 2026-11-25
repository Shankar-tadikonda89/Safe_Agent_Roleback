"""Versioned synthetic mechanisms for the Phase 9.2 offline study.

The policy sees only an ``ObservablePrefix`` converted to the fixed feature
allowlist. Scenario truth and paired outcomes remain evaluator-only data.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_demo import FEATURE_NAMES

EXPERIMENT_VERSION = "phase9-2-diverse-mechanisms-v1"
SCHEMA_VERSION = 1
CLASS_NAMES = (
    "recovery", "unnecessary_intervention", "ineffective_intervention", "intervention_harm"
)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
POLICY_NAMES = (
    "no_intervention", "naive_threshold", "proposed", "pilot_test", "prefix_branching_approximation"
)
FEATURE_ORDER = tuple(FEATURE_NAMES)
if FEATURE_ORDER != ("failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history"):
    raise RuntimeError("Phase 9.2 requires the locked four-feature schema")

# Keep these frozen Phase 9 assumptions unchanged.
FAILURE_PENALTY = 3.0
INTERVENTION_OVERHEAD_COST = 2.0
CONTINUE_COSTS = {
    "recovery": 3.0, "unnecessary_intervention": 0.0,
    "ineffective_intervention": 3.0, "intervention_harm": 0.0,
}
INTERVENE_COSTS = {
    "recovery": 2.0, "unnecessary_intervention": 2.0,
    "ineffective_intervention": 5.0, "intervention_harm": 5.0,
}
PRIMARY_HARM_PROBABILITY = 0.10
SENSITIVITY_HARM_PROBABILITIES = (0.0, 0.05, 0.20)
STREAM_NAMES = (
    "family_config", "latent", "fault", "observation", "action", "position",
    "history", "execution", "repair", "harm", "branch_execution", "branch_repair", "branch_harm",
)
MECHANISMS = (
    "transient_tool_fault", "stale_evidence", "irreversible_action", "late_recovery_window",
    "replanning_disruption", "conflicting_evidence", "tool_drift", "resource_budget",
)
TRAIN_ROOT_SEED = 20261201
VALIDATION_ROOT_SEED = 20261202
PILOT_ROOT_SEED = 20261203
TEST_ROOT_SEED = 20261204
TRAIN_FAMILY_COUNT = 12
VALIDATION_FAMILY_COUNT = 6
PILOT_FAMILY_COUNT = 4
TEST_FAMILY_COUNT = 30
SCENARIOS_PER_FAMILY = 256
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 90210
PREFIX_BRANCH_ROLLOUTS_PER_ACTION = 4
LOG_LOSS_EPSILON = 1e-15
PROBABILITY_SUM_TOLERANCE = 1e-8


@dataclass(frozen=True)
class FamilyConfig:
    family_id: str
    mechanism: str
    fault_prior: float
    sensor_noise_sd: float
    sensor_sensitivity: float
    sensor_false_positive_rate: float
    repair_probability: float
    late_repair_sensitivity: float
    irreversible_failure_sensitivity: float
    disruption_failure_sensitivity: float
    harm_multiplier: float
    action_mix: tuple[float, float, float]


@dataclass(frozen=True)
class ObservablePrefix:
    """Only pre-decision observations; contains no IDs, labels, or hidden truth."""
    noisy_risk_signal: float
    anomaly_alert: bool
    fault_prior: float
    action_kind: str
    current_step: int
    planned_steps: int
    prior_replans: int


@dataclass(frozen=True)
class Scenario:
    family: FamilyConfig
    root_seed: int
    seed: int
    harm_probability: float
    prefix: ObservablePrefix
    features: dict[str, float]
    latent_fragility: float
    stale_evidence: bool
    conflicting_evidence: bool
    fault_enabled: bool
    execution_uniform: float
    repair_uniform: float
    harm_uniform: float
    initial_state: dict[str, Any]
    scenario_hash: str


def _seed(root_seed: int, family_id: str, scenario_seed: int, stream: str, extra: int = 0) -> int:
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown stream: {stream}")
    payload = json.dumps(
        [EXPERIMENT_VERSION, int(root_seed), family_id, int(scenario_seed), stream, int(extra)],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng(root_seed: int, family_id: str, scenario_seed: int, stream: str, extra: int = 0) -> random.Random:
    return random.Random(_seed(root_seed, family_id, scenario_seed, stream, extra))


def clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def family_configurations() -> dict[str, list[FamilyConfig]]:
    """Return explicit split configurations; held-out mechanisms include unseen types."""
    spec = {
        "training": (TRAIN_FAMILY_COUNT, TRAIN_ROOT_SEED,
                     MECHANISMS[:5]),
        "validation": (VALIDATION_FAMILY_COUNT, VALIDATION_ROOT_SEED,
                       (MECHANISMS[0], MECHANISMS[1], MECHANISMS[2], MECHANISMS[4], MECHANISMS[5], MECHANISMS[6])),
        "pilot": (PILOT_FAMILY_COUNT, PILOT_ROOT_SEED,
                  (MECHANISMS[0], MECHANISMS[2], MECHANISMS[4], MECHANISMS[7])),
        "test": (TEST_FAMILY_COUNT, TEST_ROOT_SEED,
                 tuple(MECHANISMS[i % len(MECHANISMS)] for i in range(TEST_FAMILY_COUNT))),
    }
    result: dict[str, list[FamilyConfig]] = {}
    for split, (count, root, mechanisms) in spec.items():
        rows = []
        for index in range(count):
            family_id = f"p92_{split}_{index:02d}"
            rng = _rng(root, family_id, index, "family_config")
            mix = [rng.uniform(0.2, 0.5), rng.uniform(0.25, 0.55), 0.0]
            mix[2] = max(0.05, 1.0 - mix[0] - mix[1])
            total = sum(mix)
            rows.append(FamilyConfig(
                family_id=family_id,
                mechanism=mechanisms[index % len(mechanisms)],
                fault_prior=rng.uniform(0.08, 0.30),
                sensor_noise_sd=rng.uniform(0.10, 0.34),
                sensor_sensitivity=rng.uniform(0.62, 0.94),
                sensor_false_positive_rate=rng.uniform(0.04, 0.22),
                repair_probability=rng.uniform(0.35, 0.90),
                late_repair_sensitivity=rng.uniform(0.10, 0.55),
                irreversible_failure_sensitivity=rng.uniform(0.05, 0.38),
                disruption_failure_sensitivity=rng.uniform(0.05, 0.35),
                harm_multiplier=rng.uniform(0.60, 1.80),
                action_mix=tuple(x / total for x in mix),
            ))
        result[split] = rows
    return result


REVERSIBILITY = {"read_only": 0.95, "reversible_write": 0.65, "irreversible_write": 0.20}


def features_from_observation(prefix: ObservablePrefix) -> dict[str, float]:
    """Map only visible prefix fields into the fixed ordered inference schema."""
    if (prefix.action_kind not in REVERSIBILITY or prefix.planned_steps <= 0
            or not 0 <= prefix.current_step < prefix.planned_steps
            or prefix.prior_replans < 0):
        raise ValueError("invalid observable prefix")
    if not (math.isfinite(prefix.noisy_risk_signal) and 0 <= prefix.noisy_risk_signal <= 1
            and math.isfinite(prefix.fault_prior) and 0 <= prefix.fault_prior <= 1):
        raise ValueError("invalid observable risk signal")
    failure_prob = clip(
        0.03 + 0.28 * prefix.noisy_risk_signal + 0.30 * float(prefix.anomaly_alert)
        + 0.30 * prefix.fault_prior
    )
    result = {
        "failure_prob": failure_prob,
        "reversibility_proxy": REVERSIBILITY[prefix.action_kind],
        "trajectory_position": prefix.current_step / max(1, prefix.planned_steps - 1),
        "disruption_history": min(1.0, prefix.prior_replans / 3.0),
    }
    if tuple(result) != FEATURE_ORDER:
        raise AssertionError("feature order changed")
    return result


def generate_scenario(
    family: FamilyConfig,
    seed: int,
    root_seed: int,
    harm_probability: float = PRIMARY_HARM_PROBABILITY,
) -> Scenario:
    if seed < 0 or not math.isfinite(harm_probability) or not 0 <= harm_probability <= 1:
        raise ValueError("scenario seed or harm probability is invalid")
    latent = _rng(root_seed, family.family_id, seed, "latent").random()
    stale = family.mechanism == "stale_evidence" and _rng(root_seed, family.family_id, seed, "observation", 1).random() < 0.38
    conflict = family.mechanism == "conflicting_evidence" and _rng(root_seed, family.family_id, seed, "observation", 2).random() < 0.42
    signal_state = latent >= 0.55 or stale or conflict
    sensor_rng = _rng(root_seed, family.family_id, seed, "observation")
    alert = sensor_rng.random() < (family.sensor_sensitivity if signal_state else family.sensor_false_positive_rate)
    noisy = clip(sensor_rng.gauss(latent, family.sensor_noise_sd))
    action_rng = _rng(root_seed, family.family_id, seed, "action")
    r = action_rng.random()
    if r < family.action_mix[0]:
        action_kind = "read_only"
    elif r < family.action_mix[0] + family.action_mix[1]:
        action_kind = "reversible_write"
    else:
        action_kind = "irreversible_write"
    position_rng = _rng(root_seed, family.family_id, seed, "position")
    planned_steps = position_rng.randint(2, 8)
    current_step = position_rng.randrange(planned_steps)
    history_rng = _rng(root_seed, family.family_id, seed, "history")
    prior_replans = history_rng.choices((0, 1, 2, 3), weights=(0.48, 0.30, 0.16, 0.06))[0]
    prefix = ObservablePrefix(
        noisy_risk_signal=noisy, anomaly_alert=alert, fault_prior=family.fault_prior,
        action_kind=action_kind, current_step=current_step, planned_steps=planned_steps,
        prior_replans=prior_replans,
    )
    features = features_from_observation(prefix)
    fault_rate = family.fault_prior
    if family.mechanism == "tool_drift":
        fault_rate = clip(fault_rate + 0.10 * prior_replans / 3.0)
    fault = _rng(root_seed, family.family_id, seed, "fault").random() < fault_rate
    execution = _rng(root_seed, family.family_id, seed, "execution").random()
    repair = _rng(root_seed, family.family_id, seed, "repair").random()
    harm_uniform = _rng(root_seed, family.family_id, seed, "harm").random()
    initial_state = {"completed_actions": [], "visible_prefix_step": current_step}
    payload = {
        "version": EXPERIMENT_VERSION, "family": asdict(family), "root_seed": root_seed,
        "seed": seed, "harm_probability": harm_probability,
        "features": features, "initial_state": initial_state,
        "latent_fragility": latent, "fault_enabled": fault, "execution": execution,
        "repair": repair, "harm_uniform": harm_uniform,
    }
    scenario_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return Scenario(
        family=family, root_seed=root_seed, seed=seed, harm_probability=harm_probability,
        prefix=prefix, features=features, latent_fragility=latent, stale_evidence=stale,
        conflicting_evidence=conflict, fault_enabled=fault, execution_uniform=execution,
        repair_uniform=repair, harm_uniform=harm_uniform, initial_state=initial_state,
        scenario_hash=scenario_hash,
    )


def _clamp_risk(value: float) -> float:
    return clip(value)


def potential_outcomes(scenario: Scenario) -> dict[str, Any]:
    """Evaluator-only paired outcomes; this function is not called by inference."""
    f, x, p = scenario.family, scenario.features, scenario.prefix
    rev, pos, disrupt, latent = x["reversibility_proxy"], x["trajectory_position"], x["disruption_history"], scenario.latent_fragility
    continue_risk = _clamp_risk(0.04 + 0.66 * latent)
    intervene_risk = _clamp_risk(0.02 + 0.28 * latent)
    repair_probability = f.repair_probability
    harm_multiplier = f.harm_multiplier
    if f.mechanism == "stale_evidence":
        continue_risk = _clamp_risk(continue_risk + (0.28 if scenario.stale_evidence else 0.0))
        intervene_risk = _clamp_risk(intervene_risk + (0.10 if scenario.stale_evidence else 0.0))
    elif f.mechanism == "irreversible_action":
        intervene_risk = _clamp_risk(intervene_risk + f.irreversible_failure_sensitivity * (1.0 - rev))
        harm_multiplier *= 1.0 + 1.8 * (1.0 - rev)
    elif f.mechanism == "late_recovery_window":
        continue_risk = _clamp_risk(continue_risk + 0.12 * pos)
        intervene_risk = _clamp_risk(intervene_risk + 0.20 * pos)
        repair_probability = clip(repair_probability - f.late_repair_sensitivity * pos)
    elif f.mechanism == "replanning_disruption":
        intervene_risk = _clamp_risk(intervene_risk + f.disruption_failure_sensitivity * disrupt)
        harm_multiplier *= 1.0 + 1.5 * disrupt
    elif f.mechanism == "conflicting_evidence":
        continue_risk = _clamp_risk(continue_risk + (0.24 if scenario.conflicting_evidence else 0.0))
        intervene_risk = _clamp_risk(intervene_risk + (0.16 if scenario.conflicting_evidence else 0.0))
    elif f.mechanism == "tool_drift":
        continue_risk = _clamp_risk(continue_risk + 0.10 * disrupt)
        repair_probability = clip(repair_probability - 0.18 * disrupt)
    elif f.mechanism == "resource_budget":
        intervene_risk = _clamp_risk(intervene_risk + 0.25 * pos)
        repair_probability = clip(repair_probability - 0.20 * pos)
    harm_rate = clip(scenario.harm_probability * harm_multiplier * (1.0 + 0.5 * disrupt))
    harm_event = scenario.harm_uniform < harm_rate
    continue_pass = (not scenario.fault_enabled) and scenario.execution_uniform >= continue_risk
    repair_ok = (not scenario.fault_enabled) or scenario.repair_uniform < repair_probability
    budget_ok = not (f.mechanism == "resource_budget" and p.planned_steps - p.current_step <= 1)
    pre_harm_intervene_pass = repair_ok and budget_ok and scenario.execution_uniform >= intervene_risk
    intervene_pass = pre_harm_intervene_pass and not harm_event
    outcome = (
        "recovery" if not continue_pass and intervene_pass else
        "unnecessary_intervention" if continue_pass and intervene_pass else
        "ineffective_intervention" if not continue_pass and not intervene_pass else
        "intervention_harm"
    )
    return {
        "continue_passed": bool(continue_pass), "intervene_passed": bool(intervene_pass),
        "repair_succeeded": bool(repair_ok) if scenario.fault_enabled else None,
        "harm_event": bool(harm_event), "harm_probability_effective": harm_rate,
        "pre_harm_intervention_passed": bool(pre_harm_intervene_pass),
        "outcome_class": outcome,
        "continue_realized_cost": CONTINUE_COSTS[outcome],
        "intervene_realized_cost": INTERVENE_COSTS[outcome],
        "cost_components": {"failure_penalty": FAILURE_PENALTY,
                             "intervention_overhead_cost": INTERVENTION_OVERHEAD_COST,
                             "overhead_included_once": True},
    }


def prefix_branch_rollout_passes(
    scenario: Scenario, action: str, rollout_index: int, harm_probability: float
) -> bool:
    """One internal offline forecast rollout for the branching approximation.

    Its branch-specific streams are separate from the evaluator's realized
    outcome streams. This baseline has access to simulator state and is not a
    deployable inference policy.
    """
    if action not in {"continue", "intervene"}:
        raise ValueError("branch action must be continue or intervene")
    f, x, p = scenario.family, scenario.features, scenario.prefix
    rev, pos, disrupt, latent = x["reversibility_proxy"], x["trajectory_position"], x["disruption_history"], scenario.latent_fragility
    c_risk = _clamp_risk(0.04 + 0.66 * latent)
    i_risk = _clamp_risk(0.02 + 0.28 * latent)
    repair_probability, harm_multiplier = f.repair_probability, f.harm_multiplier
    if f.mechanism == "stale_evidence":
        c_risk = _clamp_risk(c_risk + (0.28 if scenario.stale_evidence else 0.0))
        i_risk = _clamp_risk(i_risk + (0.10 if scenario.stale_evidence else 0.0))
    elif f.mechanism == "irreversible_action":
        i_risk = _clamp_risk(i_risk + f.irreversible_failure_sensitivity * (1.0 - rev))
        harm_multiplier *= 1.0 + 1.8 * (1.0 - rev)
    elif f.mechanism == "late_recovery_window":
        c_risk = _clamp_risk(c_risk + 0.12 * pos)
        i_risk = _clamp_risk(i_risk + 0.20 * pos)
        repair_probability = clip(repair_probability - f.late_repair_sensitivity * pos)
    elif f.mechanism == "replanning_disruption":
        i_risk = _clamp_risk(i_risk + f.disruption_failure_sensitivity * disrupt)
        harm_multiplier *= 1.0 + 1.5 * disrupt
    elif f.mechanism == "conflicting_evidence":
        c_risk = _clamp_risk(c_risk + (0.24 if scenario.conflicting_evidence else 0.0))
        i_risk = _clamp_risk(i_risk + (0.16 if scenario.conflicting_evidence else 0.0))
    elif f.mechanism == "tool_drift":
        c_risk = _clamp_risk(c_risk + 0.10 * disrupt)
        repair_probability = clip(repair_probability - 0.18 * disrupt)
    elif f.mechanism == "resource_budget":
        i_risk = _clamp_risk(i_risk + 0.25 * pos)
        repair_probability = clip(repair_probability - 0.20 * pos)
    execution = _rng(scenario.root_seed, f.family_id, scenario.seed, "branch_execution", rollout_index).random()
    if action == "continue":
        return not scenario.fault_enabled and execution >= c_risk
    repair = _rng(scenario.root_seed, f.family_id, scenario.seed, "branch_repair", rollout_index).random()
    harm = _rng(scenario.root_seed, f.family_id, scenario.seed, "branch_harm", rollout_index).random()
    repair_ok = not scenario.fault_enabled or repair < repair_probability
    budget_ok = not (f.mechanism == "resource_budget" and p.planned_steps - p.current_step <= 1)
    effective_harm = clip(harm_probability * harm_multiplier * (1.0 + 0.5 * disrupt))
    return repair_ok and budget_ok and execution >= i_risk and harm >= effective_harm


def scenario_key(scenario: Scenario) -> dict[str, Any]:
    return {"family_id": scenario.family.family_id, "mechanism": scenario.family.mechanism,
            "root_seed": scenario.root_seed, "seed": scenario.seed,
            "scenario_hash": scenario.scenario_hash}


def family_config_record(config: FamilyConfig) -> dict[str, Any]:
    return asdict(config)


def split_manifest() -> dict[str, Any]:
    splits = family_configurations()
    roots = {"training": TRAIN_ROOT_SEED, "validation": VALIDATION_ROOT_SEED,
             "pilot": PILOT_ROOT_SEED, "test": TEST_ROOT_SEED}
    counts = {"training": TRAIN_FAMILY_COUNT, "validation": VALIDATION_FAMILY_COUNT,
              "pilot": PILOT_FAMILY_COUNT, "test": TEST_FAMILY_COUNT}
    records = {}
    for split, families in splits.items():
        records[split] = {
            "root_seed": roots[split],
            "family_configurations": [family_config_record(x) for x in families],
            "scenario_keys": [
                {"family_id": family.family_id, "root_seed": roots[split], "seed": seed}
                for family in families for seed in range(SCENARIOS_PER_FAMILY)
            ],
            "scenario_count": counts[split] * SCENARIOS_PER_FAMILY,
        }
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "feature_order": list(FEATURE_ORDER),
        "class_order": list(CLASS_NAMES),
        "costs": {"continue": CONTINUE_COSTS, "intervene": INTERVENE_COSTS},
        "harm_probabilities": [0.0, 0.05, PRIMARY_HARM_PROBABILITY, 0.20],
        "parameter_ranges": {
            "fault_prior": [0.08, 0.30], "sensor_noise_sd": [0.10, 0.34],
            "sensor_sensitivity": [0.62, 0.94], "sensor_false_positive_rate": [0.04, 0.22],
            "repair_probability": [0.35, 0.90], "late_repair_sensitivity": [0.10, 0.55],
            "irreversible_failure_sensitivity": [0.05, 0.38],
            "disruption_failure_sensitivity": [0.05, 0.35], "harm_multiplier": [0.60, 1.80],
            "read_only_action_weight": [0.20, 0.50],
            "reversible_action_weight": [0.25, 0.55],
        },
        "mechanism_rules": {
            "transient_tool_fault": "fault blocks continuation; intervention can repair at family probability",
            "stale_evidence": "stale evidence raises both branch risks, more for continuation",
            "irreversible_action": "irreversible action lowers intervention reliability and increases harm multiplier",
            "late_recovery_window": "later position raises branch risk and reduces repair probability",
            "replanning_disruption": "prior replans raise intervention risk and harm probability",
            "conflicting_evidence": "conflicting evidence raises both risks, more for continuation",
            "tool_drift": "prior replans raise fault prior and reduce intervention repair probability",
            "resource_budget": "late position raises intervention risk and can exhaust branch budget",
        },
        "family_design": {"mechanisms": list(MECHANISMS),
                          "test_includes_unseen_training_mechanisms": sorted(
                              {x.mechanism for x in splits["test"]} - {x.mechanism for x in splits["training"]})},
        "scenarios_per_family": SCENARIOS_PER_FAMILY,
        "splits": records,
    }
    all_ids = [x.family_id for configs in splits.values() for x in configs]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("family configurations overlap between splits")
    return manifest

