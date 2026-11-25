"""Four-class, cost-sensitive offline fragility experiment (Phase 9).

This module is intentionally isolated from the historical Phase 8.3 runner.
Counterfactual outcomes are generated only by the offline evaluator and are
never passed to the policy's inference method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_demo import FEATURE_NAMES, NaiveThresholdPolicy, NoInterventionPolicy


EXPERIMENT_VERSION = "phase9-four-class-cost-v1"
SCHEMA_VERSION = 1
CLASS_NAMES = (
    "recovery",
    "unnecessary_intervention",
    "ineffective_intervention",
    "intervention_harm",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
POLICY_NAMES = ("no_intervention", "naive_threshold", "proposed")
TRAIN_FAMILIES = tuple(f"phase9_train_family_{i:02d}" for i in range(12))
EVALUATION_FAMILIES = tuple(f"phase9_eval_family_{i:02d}" for i in range(20))
TRAIN_ROOT_SEED = 20261028
EVALUATION_ROOT_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
REPLICATES_PER_FAMILY = 128
PRIMARY_HARM_PROBABILITY = 0.10
SENSITIVITY_HARM_PROBABILITIES = (0.0, 0.05, 0.20)
PROBABILITY_SUM_TOLERANCE = 1e-8
LOG_LOSS_EPSILON = 1e-15
RELIABILITY_BIN_COUNT = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 90210
FAILURE_PENALTY = 3.0
INTERVENTION_OVERHEAD_COST = 2.0
INTERVENTION_OVERHEAD_STEPS_TOTAL = 2
INTERVENTION_OVERHEAD_STEPS_INCREMENTAL = 1
FAULT_PROBABILITY = 0.18
SENSOR_NOISE_SD = 0.18
FAULT_REPAIR_PROBABILITY = 0.65
BASE_FAILURE_INTERCEPT = 0.04
BASE_FAILURE_SLOPE = 0.76
INTERVENTION_FAILURE_INTERCEPT = 0.02
INTERVENTION_FAILURE_SLOPE = 0.35
STREAM_NAMES = (
    "fragility", "sensor", "fault", "execution", "fault_repair", "intervention_harm"
)

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

SCENARIO_DESIGN = """# Phase 9 scenario design

This is a synthetic offline experiment. Every evaluation scenario has two
simulator-generated potential outcomes; they are counterfactual outputs, not
simultaneously observed real-world branches.

## Outcome mechanism

Each scenario uses SHA-256-derived, deterministic, domain-separated streams
for latent fragility, sensor noise, fault injection, execution, repair, and
intervention harm. Scenario identity includes experiment version, task family,
root seed, and replicate seed. The harm draw is generated for every scenario,
even when the selected policy does not intervene. The preregistered primary
harm probability is 0.10, an illustrative assumption rather than an empirical
estimate.

Continuation passes iff no fault is injected and a shared execution draw
clears `0.04 + 0.76 * latent_fragility`. Intervention passes iff repair is
unneeded or succeeds at probability 0.65, the same execution draw clears
`0.02 + 0.35 * latent_fragility`, and the independent harm event does not
occur. The independent harm event may turn an otherwise passing continuation
into a failed intervention. These rules make recovery, unnecessary,
ineffective, and harmful outcomes possible.

## Features and training

The exact ordered feature allowlist is `failure_prob`, `reversibility_proxy`,
`trajectory_position`, `disruption_history`. Only `failure_prob` varies in
this one-step synthetic setting. Fault flags, latent values, random draws,
labels, potential outcomes, scenario hashes, IDs, and seeds are excluded from
inference features. Training examples derive four-class labels from paired
branches on training families only. The proposed inference method receives
only one four-value feature row, calls `predict_proba` once, validates class
probabilities as finite, bounded, and summing to one within absolute tolerance
`1e-8`, and minimizes expected frozen cost.

## Frozen outcome costs

Class order: recovery, unnecessary intervention, ineffective intervention,
intervention harm. Continue costs are 3, 0, 3, 0; intervene costs are 2, 2,
5, 5. Failure penalty is 3 and intervention overhead is 2, included exactly
once. An abort is a failed task at cost 3 and zero executed steps. Intervention
uses two total steps (replan plus action), one incremental overhead step.

## Split, comparison, and limits

Training uses 12 disjoint family IDs and 128 examples per family. Held-out
evaluation uses 20 different family IDs, 128 replicates each, and five fixed
root seeds. All three policies see matched scenarios and isolated initial
states. Decisions are fixed before evaluation-only potential outcomes are
written to a separate JSONL file. Families are grouping IDs from one generator,
not distinct real-world domains. The harm probability is synthetic. Counts are
not a power calculation; all findings are exploratory and do not validate
real LLM-agent performance.
"""


class InvalidOutcomeProbabilities(ValueError):
    """Raised when a multiclass estimator returns invalid probabilities."""


@dataclass(frozen=True)
class Scenario:
    task_family: str
    seed: int
    root_seed: int
    harm_probability: float
    latent_fragility: float
    noisy_fragility_signal: float
    failure_prob: float
    fault_enabled: bool
    execution_uniform: float
    repair_uniform: float
    harm_uniform: float
    harm_event: bool
    scenario_hash: str
    initial_state_hash: str


def _stream_seed(root_seed: int, family: str, seed: int, stream: str) -> int:
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown random stream: {stream}")
    material = json.dumps(
        [EXPERIMENT_VERSION, int(root_seed), family, int(seed), stream],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _rng(root_seed: int, family: str, seed: int, stream: str) -> random.Random:
    return random.Random(_stream_seed(root_seed, family, seed, stream))


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def feature_vector(failure_prob: float) -> dict[str, float]:
    if isinstance(failure_prob, bool) or not isinstance(failure_prob, (int, float)):
        raise ValueError("failure_prob must be numeric")
    if not math.isfinite(failure_prob) or not 0.0 <= failure_prob <= 1.0:
        raise ValueError("failure_prob must be finite and in [0, 1]")
    result = {
        "failure_prob": float(failure_prob),
        "reversibility_proxy": 0.25,
        "trajectory_position": 0.0,
        "disruption_history": 0.0,
    }
    if tuple(result) != FEATURE_NAMES:
        raise AssertionError("Phase 9 feature order differs from the locked project schema")
    return result


def generate_scenario(
    task_family: str,
    seed: int,
    root_seed: int,
    harm_probability: float = PRIMARY_HARM_PROBABILITY,
) -> Scenario:
    if (not task_family or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(root_seed, bool) or not isinstance(root_seed, int) or seed < 0):
        raise ValueError("task family and nonnegative integer seed are required")
    if not math.isfinite(harm_probability) or not 0.0 <= harm_probability <= 1.0:
        raise ValueError("harm_probability must be finite and in [0, 1]")
    latent = _rng(root_seed, task_family, seed, "fragility").random()
    sensor_rng = _rng(root_seed, task_family, seed, "sensor")
    observed_fragility = _clip_probability(sensor_rng.gauss(latent, SENSOR_NOISE_SD))
    baseline_risk = BASE_FAILURE_INTERCEPT + BASE_FAILURE_SLOPE * observed_fragility
    failure_prob = 1.0 - (1.0 - baseline_risk) * (1.0 - FAULT_PROBABILITY)
    fault_enabled = _rng(root_seed, task_family, seed, "fault").random() < FAULT_PROBABILITY
    execution_uniform = _rng(root_seed, task_family, seed, "execution").random()
    repair_uniform = _rng(root_seed, task_family, seed, "fault_repair").random()
    # This draw occurs for every scenario, independent of which policy acts.
    harm_uniform = _rng(root_seed, task_family, seed, "intervention_harm").random()
    harm_event = harm_uniform < harm_probability
    initial_state = {"completed_actions": []}
    initial_state_hash = hashlib.sha256(
        json.dumps(initial_state, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "experiment": EXPERIMENT_VERSION,
        "task_family": task_family,
        "seed": int(seed),
        "root_seed": int(root_seed),
        "harm_probability": harm_probability,
        "latent_fragility": latent,
        "noisy_fragility_signal": observed_fragility,
        "failure_prob": failure_prob,
        "fault_enabled": fault_enabled,
        "execution_uniform": execution_uniform,
        "repair_uniform": repair_uniform,
        "harm_uniform": harm_uniform,
    }
    scenario_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return Scenario(
        task_family=task_family,
        seed=int(seed),
        root_seed=int(root_seed),
        harm_probability=float(harm_probability),
        latent_fragility=latent,
        noisy_fragility_signal=observed_fragility,
        failure_prob=failure_prob,
        fault_enabled=fault_enabled,
        execution_uniform=execution_uniform,
        repair_uniform=repair_uniform,
        harm_uniform=harm_uniform,
        harm_event=harm_event,
        scenario_hash=scenario_hash,
        initial_state_hash=initial_state_hash,
    )


def potential_outcomes(scenario: Scenario) -> dict[str, Any]:
    """Offline evaluator only; never called by the policy inference method."""
    continue_risk = BASE_FAILURE_INTERCEPT + BASE_FAILURE_SLOPE * scenario.latent_fragility
    continue_passed = (
        not scenario.fault_enabled and scenario.execution_uniform >= continue_risk
    )
    intervene_risk = (
        INTERVENTION_FAILURE_INTERCEPT
        + INTERVENTION_FAILURE_SLOPE * scenario.latent_fragility
    )
    repair_succeeded = (
        None if not scenario.fault_enabled
        else scenario.repair_uniform < FAULT_REPAIR_PROBABILITY
    )
    repair_ok = repair_succeeded is not False
    pre_harm_intervention_pass = (
        repair_ok and scenario.execution_uniform >= intervene_risk
    )
    intervene_passed = pre_harm_intervention_pass and not scenario.harm_event
    if not continue_passed and intervene_passed:
        outcome = "recovery"
    elif continue_passed and intervene_passed:
        outcome = "unnecessary_intervention"
    elif not continue_passed and not intervene_passed:
        outcome = "ineffective_intervention"
    else:
        outcome = "intervention_harm"
    return {
        "continue_passed": bool(continue_passed),
        "intervene_passed": bool(intervene_passed),
        "repair_status": (
            "not_needed" if repair_succeeded is None
            else "repaired" if repair_succeeded else "failed"
        ),
        "repair_succeeded": repair_succeeded,
        "harm_event": bool(scenario.harm_event),
        "harm_probability": scenario.harm_probability,
        "pre_harm_intervention_passed": bool(pre_harm_intervention_pass),
        "outcome_class": outcome,
        "continue_realized_cost": CONTINUE_COSTS[outcome],
        "intervene_realized_cost": INTERVENE_COSTS[outcome],
        "cost_components": {
            "failure_penalty": FAILURE_PENALTY,
            "intervention_overhead_cost": INTERVENTION_OVERHEAD_COST,
            "overhead_included_in_intervention_cost": True,
            "additional_harm_surcharge": 0.0,
        },
        "intervention_overhead_steps_total": INTERVENTION_OVERHEAD_STEPS_TOTAL,
        "intervention_overhead_steps_incremental": INTERVENTION_OVERHEAD_STEPS_INCREMENTAL,
    }


def _category_index(continue_passed: bool, intervene_passed: bool) -> int:
    if not continue_passed and intervene_passed:
        return CLASS_TO_INDEX["recovery"]
    if continue_passed and intervene_passed:
        return CLASS_TO_INDEX["unnecessary_intervention"]
    if not continue_passed and not intervene_passed:
        return CLASS_TO_INDEX["ineffective_intervention"]
    return CLASS_TO_INDEX["intervention_harm"]


def _scenario_key(scenario: Scenario) -> tuple[str, int, int]:
    return scenario.task_family, scenario.root_seed, scenario.seed


def _public_scenario_key(scenario: Scenario) -> dict[str, Any]:
    return {
        "task_family": scenario.task_family,
        "root_seed": scenario.root_seed,
        "seed": scenario.seed,
        "scenario_hash": scenario.scenario_hash,
    }


def _features(scenario: Scenario) -> dict[str, float]:
    return feature_vector(scenario.failure_prob)


def _make_training_example(scenario: Scenario) -> dict[str, Any]:
    # Training labels are generated from paired branches in the training split.
    outcomes = potential_outcomes(scenario)
    label = _category_index(outcomes["continue_passed"], outcomes["intervene_passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "scenario_key": _public_scenario_key(scenario),
        "feature_names": list(FEATURE_NAMES),
        "features": _features(scenario),
        "outcome_class_index": label,
        "outcome_class": CLASS_NAMES[label],
        "potential_outcomes": outcomes,
        "scenario_metadata": {
            "latent_fragility": scenario.latent_fragility,
            "noisy_fragility_signal": scenario.noisy_fragility_signal,
            "fault_enabled": scenario.fault_enabled,
        },
    }


def generate_training_examples(
    root_seed: int = TRAIN_ROOT_SEED,
    replicates_per_family: int = REPLICATES_PER_FAMILY,
    families: Iterable[str] = TRAIN_FAMILIES,
    harm_probability: float = PRIMARY_HARM_PROBABILITY,
) -> list[dict[str, Any]]:
    family_ids = tuple(families)
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise ValueError("training family IDs must be non-empty and unique")
    if set(family_ids) & set(EVALUATION_FAMILIES):
        raise ValueError("training and held-out family IDs must be disjoint")
    if replicates_per_family <= 0:
        raise ValueError("replicates_per_family must be positive")
    return [
        _make_training_example(
            generate_scenario(family, seed, root_seed, harm_probability)
        )
        for family in family_ids
        for seed in range(replicates_per_family)
    ]


def train_model(
    examples: list[dict[str, Any]],
    training_families: Iterable[str] = TRAIN_FAMILIES,
    harm_probability: float = PRIMARY_HARM_PROBABILITY,
) -> tuple[Pipeline, dict[str, Any]]:
    family_ids = tuple(training_families)
    if not examples:
        raise ValueError("training examples cannot be empty")
    if set(family_ids) & set(EVALUATION_FAMILIES):
        raise ValueError("training families overlap held-out families")
    if any(row["scenario_key"]["task_family"] not in family_ids for row in examples):
        raise ValueError("held-out or undeclared family entered training")
    if any(row.get("feature_names") != list(FEATURE_NAMES) for row in examples):
        raise ValueError("training feature schema/order is invalid")
    labels = np.asarray([row["outcome_class_index"] for row in examples], dtype=int)
    expected_classes = np.arange(len(CLASS_NAMES))
    if not np.array_equal(np.unique(labels), expected_classes):
        missing = [CLASS_NAMES[index] for index in expected_classes if index not in labels]
        raise ValueError(f"training data is missing required outcome classes: {missing}")
    x = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in examples],
        dtype=float,
    )
    if x.shape != (len(examples), len(FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError("training features have invalid shape or values")
    # No class weighting: retain the natural simulated class prevalence for
    # probability interpretation. Solver and seed are fixed for reproduction.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(solver="lbfgs", random_state=0, max_iter=2000)),
    ])
    model.fit(x, labels)
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError("fitted classifier class ordering is incomplete or unexpected")
    counts = {name: int(np.sum(labels == index)) for index, name in enumerate(CLASS_NAMES)}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "costs": {
            "continue": CONTINUE_COSTS,
            "intervene": INTERVENE_COSTS,
            "failure_penalty": FAILURE_PENALTY,
            "intervention_overhead_cost": INTERVENTION_OVERHEAD_COST,
            "intervention_overhead_included": True,
            "additional_harm_surcharge": 0.0,
        },
        "training_families": list(family_ids),
        "evaluation_families": list(EVALUATION_FAMILIES),
        "training_root_seed": TRAIN_ROOT_SEED,
        "training_replicates_per_family": REPLICATES_PER_FAMILY,
        "training_harm_probability": harm_probability,
        "training_class_counts": counts,
        "model": {
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "solver": "lbfgs",
            "random_state": 0,
            "max_iter": 2000,
            "class_weight": None,
        },
        "limitations": [
            "Synthetic outcomes and illustrative harm probability; not empirical LLM-agent evidence.",
            "Task family IDs share one generator and are not independent real-world domains.",
            "Only failure_prob varies among the four allowed one-step features.",
        ],
    }
    return model, metadata


def _validate_probability_vector(probabilities: Any) -> np.ndarray:
    try:
        raw = np.asarray(probabilities, dtype=object)
    except (TypeError, ValueError) as exc:
        raise InvalidOutcomeProbabilities("probabilities must be numeric") from exc
    if raw.shape != (len(CLASS_NAMES),):
        raise InvalidOutcomeProbabilities("probability vector has an invalid shape")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        raise InvalidOutcomeProbabilities("probabilities must contain numeric values")
    vector = raw.astype(float)
    if not np.isfinite(vector).all():
        raise InvalidOutcomeProbabilities("probabilities must be finite")
    if ((vector < 0.0) | (vector > 1.0)).any():
        raise InvalidOutcomeProbabilities("probabilities must be within [0, 1]")
    if not math.isclose(float(vector.sum()), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise InvalidOutcomeProbabilities("probabilities must sum to one within tolerance")
    return vector


def calculate_expected_costs(probabilities: Any) -> dict[str, Any]:
    q = _validate_probability_vector(probabilities)
    q_r, _q_u, q_n, q_h = (float(value) for value in q)
    continue_cost = 3.0 * (q_r + q_n)
    intervene_cost = 2.0 + 3.0 * (q_n + q_h)
    return {
        "class_names": list(CLASS_NAMES),
        "class_probabilities": [float(value) for value in q],
        "expected_continue_cost": continue_cost,
        "expected_intervene_cost": intervene_cost,
        "selected_action": "intervene" if intervene_cost < continue_cost else "continue",
    }


class ProposedPolicy:
    """Four-class policy: one model prediction and deterministic cost minimization."""

    def __init__(self, model: Pipeline):
        if not hasattr(model, "predict_proba") or not np.array_equal(
            np.asarray(getattr(model, "classes_", [])), np.arange(len(CLASS_NAMES))
        ):
            raise ValueError("model classes must be ordered exactly as the Phase 9 class indices")
        self.model = model

    def decide(self, features: dict[str, float]) -> dict[str, Any]:
        if tuple(features) != FEATURE_NAMES:
            raise ValueError("policy requires the exact ordered feature allowlist")
        x = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
        if x.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise ValueError("inference features must be one finite row in [0, 1]")
        raw = np.asarray(self.model.predict_proba(x), dtype=float)
        if raw.shape != (1, len(CLASS_NAMES)):
            raise InvalidOutcomeProbabilities("model returned an invalid probability matrix shape")
        diagnostics = calculate_expected_costs(raw[0])
        return diagnostics


def save_model(model: Pipeline, metadata: dict[str, Any], artifact_path: str | Path) -> dict[str, Path]:
    artifact_path = Path(artifact_path)
    sidecar_path = Path(str(artifact_path) + ".metadata.json")
    if artifact_path.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to overwrite an existing Phase 9 model or metadata")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact_path)
    checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    sidecar = {"artifact_sha256": checksum, "model_metadata": metadata}
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"artifact": artifact_path, "metadata": sidecar_path}


def load_model(artifact_path: str | Path) -> tuple[ProposedPolicy, dict[str, Any], str]:
    artifact_path = Path(artifact_path)
    sidecar_path = Path(str(artifact_path) + ".metadata.json")
    if not artifact_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("Phase 9 model artifact or metadata is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if checksum != sidecar.get("artifact_sha256"):
        raise ValueError("Phase 9 model artifact checksum mismatch")
    metadata = sidecar.get("model_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Phase 9 model metadata is invalid")
    _validate_metadata(metadata)
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict) or payload.get("metadata") != metadata:
        raise ValueError("Phase 9 model payload metadata mismatch")
    return ProposedPolicy(payload.get("model")), metadata, checksum


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if (metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("experiment_version") != EXPERIMENT_VERSION
            or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("class_names") != list(CLASS_NAMES)
            or metadata.get("costs", {}).get("continue") != CONTINUE_COSTS
            or metadata.get("costs", {}).get("intervene") != INTERVENE_COSTS
            or set(metadata.get("training_families", ())) & set(metadata.get("evaluation_families", ()))):
        raise ValueError("Phase 9 model metadata is incompatible")


def _ensure_fresh_output(output_dir: Path, artifact_path: Path | None = None) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing Phase 9 output: {output_dir}")
    if artifact_path is not None:
        sidecar = Path(str(artifact_path) + ".metadata.json")
        if artifact_path.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite existing Phase 9 model: {artifact_path}")


def generate_evaluation_scenarios(
    harm_probability: float,
    families: Iterable[str] = EVALUATION_FAMILIES,
    root_seeds: Iterable[int] = EVALUATION_ROOT_SEEDS,
    replicates_per_family: int = REPLICATES_PER_FAMILY,
) -> list[Scenario]:
    family_ids, roots = tuple(families), tuple(int(x) for x in root_seeds)
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise ValueError("evaluation family IDs must be non-empty and unique")
    if set(family_ids) & set(TRAIN_FAMILIES):
        raise ValueError("training and evaluation families must be disjoint")
    if not roots or len(set(roots)) != len(roots) or replicates_per_family <= 0:
        raise ValueError("evaluation roots must be unique and replicates positive")
    return [
        generate_scenario(family, seed, root, harm_probability)
        for family in family_ids
        for root in roots
        for seed in range(replicates_per_family)
    ]


def _select_decision(scenario: Scenario, policy_name: str, proposed: ProposedPolicy | None) -> dict[str, Any]:
    features = _features(scenario)
    if policy_name == "no_intervention":
        action = NoInterventionPolicy().decide(features["failure_prob"], features)
        diagnostics = None
    elif policy_name == "naive_threshold":
        action = NaiveThresholdPolicy().decide(features["failure_prob"], features)
        diagnostics = None
    elif policy_name == "proposed":
        if proposed is None:
            raise ValueError("proposed policy requires the Phase 9 model")
        diagnostics = proposed.decide(features)
        action = diagnostics["selected_action"]
    else:
        raise ValueError(f"unknown policy: {policy_name}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "scenario_key": _public_scenario_key(scenario),
        "policy": policy_name,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "decision": action,
        "decision_diagnostics": diagnostics,
        "initial_state": {"completed_actions": []},
        "initial_state_hash": scenario.initial_state_hash,
    }


def _potential_record(scenario: Scenario) -> dict[str, Any]:
    outcome = potential_outcomes(scenario)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "record_type": "evaluation_only_potential_outcomes",
        "scenario_key": _public_scenario_key(scenario),
        "potential_outcomes": outcome,
        "scenario_metadata": {
            "fault_enabled": scenario.fault_enabled,
            "latent_fragility": scenario.latent_fragility,
            "noisy_fragility_signal": scenario.noisy_fragility_signal,
        },
    }


def _run_policies_after_decisions(
    scenarios: list[Scenario],
    proposed: ProposedPolicy,
    model_checksum: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Freeze all policy decisions first. This loop receives feature-only data.
    decision_rows: list[dict[str, Any]] = []
    elapsed: dict[tuple[tuple[str, int, int], str], float] = {}
    for scenario in scenarios:
        for policy_name in POLICY_NAMES:
            started = time.perf_counter()
            row = _select_decision(
                scenario, policy_name, proposed if policy_name == "proposed" else None
            )
            elapsed[(_scenario_key(scenario), policy_name)] = time.perf_counter() - started
            decision_rows.append(row)

    # Only now does the evaluator calculate and record counterfactual outcomes.
    potential_rows = [_potential_record(scenario) for scenario in scenarios]
    potential_by_key = {
        (row["scenario_key"]["task_family"], row["scenario_key"]["root_seed"], row["scenario_key"]["seed"]): row
        for row in potential_rows
    }
    policy_rows: list[dict[str, Any]] = []
    for row in decision_rows:
        key_object = row["scenario_key"]
        key = (key_object["task_family"], key_object["root_seed"], key_object["seed"])
        potential = potential_by_key[key]["potential_outcomes"]
        action = row["decision"]
        if action == "continue":
            passed = potential["continue_passed"]
            cost = potential["continue_realized_cost"]
            steps = 1
        elif action == "intervene":
            passed = potential["intervene_passed"]
            cost = potential["intervene_realized_cost"]
            steps = INTERVENTION_OVERHEAD_STEPS_TOTAL
        elif action == "abort":
            passed, cost, steps = False, FAILURE_PENALTY, 0
        else:
            raise ValueError(f"unsupported selected action: {action}")
        row.update({
            "selected_outcome_passed": bool(passed),
            "success": bool(passed),
            "failure": not bool(passed),
            "realized_cost": float(cost),
            "steps_executed": steps,
            "intervention": action == "intervene",
            "overhead_steps": INTERVENTION_OVERHEAD_STEPS_INCREMENTAL if action == "intervene" else 0,
            "local_latency_seconds": elapsed[((key[0], key[1], key[2]), row["policy"])],
            "mode": "synthetic_offline",
            "llm_calls": 0,
            "model_sha256": model_checksum if row["policy"] == "proposed" else None,
        })
        policy_rows.append(row)
    return policy_rows, potential_rows


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def _cluster_bootstrap_interval(
    differences: dict[str, list[float]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    family_means = {
        family: statistics.mean(values) for family, values in sorted(differences.items())
    }
    if len(family_means) < 2:
        raise ValueError("cluster bootstrap requires at least two held-out families")
    rng = random.Random(seed)
    means = list(family_means.values())
    samples = [
        statistics.mean(rng.choices(means, k=len(means)))
        for _ in range(resamples)
    ]
    return {
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
        "resamples": resamples,
        "cluster": "heldout_task_family",
        "bootstrap_seed": seed,
    }


def calculate_metrics(
    policy_rows: list[dict[str, Any]],
    potential_rows: list[dict[str, Any]],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    potentials = {
        (r["scenario_key"]["task_family"], r["scenario_key"]["root_seed"], r["scenario_key"]["seed"]): r["potential_outcomes"]
        for r in potential_rows
    }
    scenario_keys = set(potentials)
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in policy_rows:
        so = row["scenario_key"]
        key = (so["task_family"], so["root_seed"], so["seed"])
        grouped.setdefault(key, {})[row["policy"]] = row
    if set(grouped) != scenario_keys or any(set(v) != set(POLICY_NAMES) for v in grouped.values()):
        raise ValueError("metrics require exactly three matched policy rows per scenario")
    by_policy: dict[str, dict[str, Any]] = {}
    control_pass = {
        key: potential["continue_passed"] for key, potential in potentials.items()
    }
    total_continue_failures = sum(not flag for flag in control_pass.values())
    total_continue_successes = sum(control_pass.values())
    for policy in POLICY_NAMES:
        rows = [grouped[key][policy] for key in sorted(grouped)]
        interventions = [row for row in rows if row["intervention"]]
        selected_categories = [
            potentials[key]["outcome_class"]
            for key in sorted(grouped)
            if grouped[key][policy]["intervention"]
        ]
        counts = {name: selected_categories.count(name) for name in CLASS_NAMES}
        passed = sum(row["success"] for row in rows)
        row_data = {
            "attempts": len(rows),
            "successes": passed,
            "failure_count": len(rows) - passed,
            "success_rate": passed / len(rows),
            "mean_realized_cost": statistics.mean(row["realized_cost"] for row in rows),
            "intervention_count": len(interventions),
            "intervention_rate": len(interventions) / len(rows),
            "abort_count": sum(row["decision"] == "abort" for row in rows),
            "recovery_count": counts["recovery"],
            "recovery_denominator": total_continue_failures,
            "recovery_rate": _rate(counts["recovery"], total_continue_failures),
            "unnecessary_intervention_count": counts["unnecessary_intervention"],
            "unnecessary_intervention_denominator": total_continue_successes,
            "unnecessary_intervention_rate": _rate(counts["unnecessary_intervention"], total_continue_successes),
            "ineffective_intervention_count": counts["ineffective_intervention"],
            "ineffective_intervention_rate_per_intervention": _rate(counts["ineffective_intervention"], len(interventions)),
            "intervention_harm_count": counts["intervention_harm"],
            "intervention_harm_rate_per_intervention": _rate(counts["intervention_harm"], len(interventions)),
            "potential_harm_count": sum(p["outcome_class"] == "intervention_harm" for p in potentials.values()),
            "potential_harm_rate": sum(p["outcome_class"] == "intervention_harm" for p in potentials.values()) / len(potentials),
            "overhead_steps": sum(row["overhead_steps"] for row in rows),
            "average_overhead_steps": statistics.mean(row["overhead_steps"] for row in rows),
            "average_steps": statistics.mean(row["steps_executed"] for row in rows),
            "average_local_runtime_seconds": statistics.mean(row["local_latency_seconds"] for row in rows),
            "selected_intervention_outcomes": counts,
        }
        by_policy[policy] = row_data

    pairwise: dict[str, Any] = {}
    for baseline in ("no_intervention", "naive_threshold"):
        diffs_by_family: dict[str, list[float]] = {}
        differences = []
        for key in sorted(grouped):
            proposed_cost = grouped[key]["proposed"]["realized_cost"]
            baseline_cost = grouped[key][baseline]["realized_cost"]
            delta = proposed_cost - baseline_cost
            differences.append(delta)
            diffs_by_family.setdefault(key[0], []).append(delta)
        pairwise[f"proposed_minus_{baseline}"] = {
            "mean_cost_difference": statistics.mean(differences),
            "confidence_interval_95": _cluster_bootstrap_interval(
                diffs_by_family, bootstrap_resamples, bootstrap_seed
            ),
        }

    proposed_rows = [grouped[key]["proposed"] for key in sorted(grouped)]
    q_rows = [_validate_probability_vector(r["decision_diagnostics"]["class_probabilities"]) for r in proposed_rows]
    true_indices = [CLASS_TO_INDEX[potentials[key]["outcome_class"]] for key in sorted(grouped)]
    predicted_indices = [int(np.argmax(q)) for q in q_rows]
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for actual, predicted in zip(true_indices, predicted_indices):
        confusion[actual, predicted] += 1
    per_class: dict[str, Any] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        predicted_positive = int(confusion[:, index].sum())
        actual_positive = int(confusion[index, :].sum())
        fp, fn = predicted_positive - tp, actual_positive - tp
        precision = tp / predicted_positive if predicted_positive else None
        recall = tp / actual_positive if actual_positive else None
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        probabilities = [float(q[index]) for q in q_rows]
        labels = [int(actual == index) for actual in true_indices]
        reliability = []
        for bin_index in range(RELIABILITY_BIN_COUNT):
            low, high = bin_index / RELIABILITY_BIN_COUNT, (bin_index + 1) / RELIABILITY_BIN_COUNT
            members = [(p, y) for p, y in zip(probabilities, labels)
                       if low <= p < high or (bin_index == RELIABILITY_BIN_COUNT - 1 and p == 1.0)]
            reliability.append({
                "bin_low": low,
                "bin_high": high,
                "count": len(members),
                "mean_predicted": statistics.mean(p for p, _ in members) if members else None,
                "observed_rate": statistics.mean(y for _, y in members) if members else None,
            })
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_predicted_probability": statistics.mean(probabilities),
            "predicted_argmax_frequency": predicted_indices.count(index) / len(predicted_indices),
            "observed_frequency": statistics.mean(labels),
            "reliability_bins": reliability,
        }

    brier = statistics.mean(
        sum((float(q[index]) - float(actual == index)) ** 2 for index in range(len(CLASS_NAMES)))
        for q, actual in zip(q_rows, true_indices)
    )
    clipped = np.maximum(np.asarray(q_rows, dtype=float), LOG_LOSS_EPSILON)
    clipped /= clipped.sum(axis=1, keepdims=True)
    log_loss = statistics.mean(-math.log(float(clipped[i, actual])) for i, actual in enumerate(true_indices))
    recovery_probabilities = [float(q[CLASS_TO_INDEX["recovery"]]) for q in q_rows]
    recovery_labels = [int(actual == CLASS_TO_INDEX["recovery"]) for actual in true_indices]
    recovery_brier = statistics.mean((p - y) ** 2 for p, y in zip(recovery_probabilities, recovery_labels))
    recovery_log_loss = statistics.mean(
        -(y * math.log(min(1.0, max(LOG_LOSS_EPSILON, p)))
          + (1-y) * math.log(min(1.0, max(LOG_LOSS_EPSILON, 1-p))))
        for p, y in zip(recovery_probabilities, recovery_labels)
    )
    outcome_counts = {name: true_indices.count(i) for i, name in enumerate(CLASS_NAMES)}
    metrics = {
        "experiment_version": EXPERIMENT_VERSION,
        "scenario_count": len(scenario_keys),
        "policy_run_count": len(policy_rows),
        "class_order": list(CLASS_NAMES),
        "cost_table": {"continue": CONTINUE_COSTS, "intervene": INTERVENE_COSTS},
        "outcome_class_counts": outcome_counts,
        "outcome_class_frequencies": {name: count / len(true_indices) for name, count in outcome_counts.items()},
        "policies": by_policy,
        "paired_cost_differences": pairwise,
        "probability_quality": {
            "multiclass_brier_score": brier,
            "multiclass_brier_formula": "mean_scenario(sum_class((p_class - one_hot_class)^2))",
            "multiclass_log_loss": log_loss,
            "log_loss_clip_epsilon": LOG_LOSS_EPSILON,
            "mean_predicted_class_probabilities": {name: per_class[name]["mean_predicted_probability"] for name in CLASS_NAMES},
            "predicted_argmax_class_frequencies": {name: per_class[name]["predicted_argmax_frequency"] for name in CLASS_NAMES},
            "observed_class_frequencies": {name: per_class[name]["observed_frequency"] for name in CLASS_NAMES},
            "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
            "per_class": per_class,
            "binary_recovery_brier_score": recovery_brier,
            "binary_recovery_log_loss": recovery_log_loss,
            "binary_recovery_mean_predicted": statistics.mean(recovery_probabilities),
            "binary_recovery_observed_rate": statistics.mean(recovery_labels),
        },
        "limitations": [
            "Synthetic potential outcomes are counterfactual simulator outputs, not simultaneous real observations.",
            "Illustrative harm rates are assumptions, not empirical estimates.",
            "Task families share one generator; the experiment is exploratory and not a power calculation.",
            "Runtime is local simulation runtime, not provider latency.",
        ],
    }
    return metrics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def _metadata_for_training(examples: list[dict[str, Any]]) -> tuple[Pipeline, dict[str, Any]]:
    return train_model(examples, TRAIN_FAMILIES, PRIMARY_HARM_PROBABILITY)


def _metric_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(_metric_equal(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_metric_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return isinstance(expected, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def _validate_record_sets(
    policy_rows: list[dict[str, Any]],
    potential_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]] | None = None,
) -> None:
    potential_by_key = {}
    seen_scenario_hashes = set()
    for row in potential_rows:
        key_obj = row["scenario_key"]
        key = (key_obj["task_family"], key_obj["root_seed"], key_obj["seed"])
        if key in potential_by_key:
            raise ValueError(f"duplicate potential-outcome scenario key: {key}")
        if key_obj["scenario_hash"] in seen_scenario_hashes:
            raise ValueError(f"duplicate potential-outcome scenario hash for {key}")
        seen_scenario_hashes.add(key_obj["scenario_hash"])
        potential = row["potential_outcomes"]
        expected_class = CLASS_NAMES[_category_index(potential["continue_passed"], potential["intervene_passed"])]
        if potential.get("outcome_class") != expected_class:
            raise ValueError(f"potential-outcome label mismatch for {key}")
        if expected_class == "intervention_harm" and not potential.get("harm_event"):
            raise ValueError("harm category requires an intervention harm event")
        if potential["intervene_passed"] != (
            potential["pre_harm_intervention_passed"] and not potential["harm_event"]
        ):
            raise ValueError(f"harm event does not reconcile with intervention outcome for {key}")
        if (potential["repair_status"] == "not_needed") != (potential["repair_succeeded"] is None):
            raise ValueError(f"repair status is inconsistent for {key}")
        if potential["repair_status"] == "repaired" and potential["repair_succeeded"] is not True:
            raise ValueError(f"repair success flag is inconsistent for {key}")
        if potential["repair_status"] == "failed" and potential["repair_succeeded"] is not False:
            raise ValueError(f"repair failure flag is inconsistent for {key}")
        if potential["continue_realized_cost"] != CONTINUE_COSTS[expected_class]:
            raise ValueError("continuation potential cost does not match frozen table")
        if potential["intervene_realized_cost"] != INTERVENE_COSTS[expected_class]:
            raise ValueError("intervention potential cost does not match frozen table")
        potential_by_key[key] = potential
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in policy_rows:
        if row.get("feature_names") != list(FEATURE_NAMES) or set(row.get("features", {})) != set(FEATURE_NAMES):
            raise ValueError("policy row has unexpected features")
        forbidden = {"fault_enabled", "latent_fragility", "scenario_hash", "task_family", "task_id", "seed",
                     "outcome_class", "potential_outcomes", "harm_event", "branches", "scenario_metadata"}
        if forbidden & set(row["features"]):
            raise ValueError("hidden metadata or outcomes entered policy features")
        key_obj = row["scenario_key"]
        key = (key_obj["task_family"], key_obj["root_seed"], key_obj["seed"])
        if row["policy"] in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate policy row for {key}/{row['policy']}")
        grouped[key][row["policy"]] = row
    if set(grouped) != set(potential_by_key):
        raise ValueError("policy and potential-outcome scenario keys differ")
    for key, policies in grouped.items():
        if set(policies) != set(POLICY_NAMES):
            raise ValueError(f"scenario {key} does not have exactly one row per policy")
        rows = list(policies.values())
        reference = rows[0]
        for row in rows:
            if row["scenario_key"] != reference["scenario_key"] or row["features"] != reference["features"]:
                raise ValueError(f"matched policy scenario mismatch for {key}")
            if row["initial_state_hash"] != reference["initial_state_hash"]:
                raise ValueError(f"initial-state hash mismatch for {key}")
            action = row["decision"]
            potential = potential_by_key[key]
            if action == "continue":
                expected_pass = potential["continue_passed"]
                expected_cost = potential["continue_realized_cost"]
            elif action == "intervene":
                expected_pass = potential["intervene_passed"]
                expected_cost = potential["intervene_realized_cost"]
            elif action == "abort":
                expected_pass, expected_cost = False, FAILURE_PENALTY
            else:
                raise ValueError(f"unknown action {action}")
            if row["selected_outcome_passed"] != expected_pass or row["realized_cost"] != expected_cost:
                raise ValueError(f"selected outcome/cost does not match potential outcome for {key}")
            if row["policy"] == "proposed":
                diagnostic = row["decision_diagnostics"]
                calculated = calculate_expected_costs(diagnostic["class_probabilities"])
                if not _metric_equal(calculated, diagnostic) or action != diagnostic["selected_action"]:
                    raise ValueError("proposed diagnostics or selected action are inconsistent")
    if training_rows is not None:
        train_keys = set()
        train_hashes = set()
        train_families = set()
        for row in training_rows:
            key_obj = row["scenario_key"]
            key = (key_obj["task_family"], key_obj["root_seed"], key_obj["seed"])
            if key in train_keys:
                raise ValueError(f"duplicate training key {key}")
            if key_obj["scenario_hash"] in train_hashes:
                raise ValueError(f"duplicate training scenario hash {key}")
            train_keys.add(key)
            train_hashes.add(key_obj["scenario_hash"])
            train_families.add(key_obj["task_family"])
            if key_obj["task_family"] in EVALUATION_FAMILIES:
                raise ValueError("held-out family found in training records")
            if row["feature_names"] != list(FEATURE_NAMES) or set(row["features"]) != set(FEATURE_NAMES):
                raise ValueError("training record feature schema is invalid")
            label = CLASS_NAMES[_category_index(
                row["potential_outcomes"]["continue_passed"],
                row["potential_outcomes"]["intervene_passed"],
            )]
            if label != row["outcome_class"]:
                raise ValueError("training label does not match paired training outcomes")
        if train_families & {key[0] for key in potential_by_key}:
            raise ValueError("training/evaluation families overlap")


def _summary_text(metrics: dict[str, Any], harm_probability: float, model_sha256: str) -> str:
    lines = [
        "# Phase 9 four-class cost-sensitive intervention experiment",
        "",
        "Synthetic offline potential-outcome simulation. Potential outcomes are counterfactual simulator outputs, not simultaneously observed real-world branches. No real LLM-agent effectiveness is established.",
        "",
        f"- Experiment: `{EXPERIMENT_VERSION}`; harm probability: `{harm_probability:.2f}` (illustrative synthetic assumption, not empirical).",
        f"- Scenarios: `{metrics['scenario_count']}`; policy runs: `{metrics['policy_run_count']}`; model SHA-256: `{model_sha256}`.",
        "- Class order: `recovery`, `unnecessary_intervention`, `ineffective_intervention`, `intervention_harm`.",
        "- Costs: continue 3/0/3/0; intervene 2/2/5/5 in the class order above. F=3, O=2; O is already included; no additional surcharge.",
        "- Primary outcome: mean realized cost. Paired cost-difference intervals are 95% family-cluster bootstrap percentile intervals with 10,000 resamples; exploratory, not a power calculation.",
        "- Runtime is local simulation runtime, not provider latency. Aborts remain failures with cost 3 and zero executed steps.",
        "- The naive threshold retains the historical rule: abort when failure_prob > 0.5, otherwise continue. Abort is scored as a task failure (3) with no intervention overhead.",
        "",
        "| Policy | Success | Failures | Mean cost | Recovery (count/rate) | Unnecessary (count/rate) | Ineffective (count/rate) | Harm (count/rate) | Interventions | Overhead steps | Avg steps | Local runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_NAMES:
        row = metrics["policies"][policy]
        recovery = "n/a" if row["recovery_rate"] is None else f"{row['recovery_count']}/{row['recovery_denominator']} ({row['recovery_rate']:.1%})"
        unnecessary = "n/a" if row["unnecessary_intervention_rate"] is None else f"{row['unnecessary_intervention_count']}/{row['unnecessary_intervention_denominator']} ({row['unnecessary_intervention_rate']:.1%})"
        ineffective = "n/a" if row["ineffective_intervention_rate_per_intervention"] is None else f"{row['ineffective_intervention_count']}/{row['intervention_count']} ({row['ineffective_intervention_rate_per_intervention']:.1%})"
        harm = "n/a" if row["intervention_harm_rate_per_intervention"] is None else f"{row['intervention_harm_count']}/{row['intervention_count']} ({row['intervention_harm_rate_per_intervention']:.1%})"
        lines.append(
            f"| {policy} | {row['successes']}/{row['attempts']} ({row['success_rate']:.1%}) | "
            f"{row['failure_count']} | {row['mean_realized_cost']:.4f} | {recovery} | {unnecessary} | "
            f"{ineffective} | {harm} | "
            f"{row['intervention_count']}/{row['attempts']} | {row['overhead_steps']} | "
            f"{row['average_steps']:.3f} | {row['average_local_runtime_seconds']:.6f} |"
        )
    lines.extend(["", "## Paired cost differences", ""])
    for label, comparison in metrics["paired_cost_differences"].items():
        ci = comparison["confidence_interval_95"]
        lines.append(
            f"- `{label}`: {comparison['mean_cost_difference']:.5f}; 95% family-cluster bootstrap interval "
            f"[{ci['lower_95']:.5f}, {ci['upper_95']:.5f}]."
        )
    quality = metrics["probability_quality"]
    lines.extend([
        "", "## Probability quality", "",
        f"- Multiclass Brier: `{quality['multiclass_brier_score']:.6f}`; multiclass log loss: `{quality['multiclass_log_loss']:.6f}` (epsilon `{quality['log_loss_clip_epsilon']}`).",
        f"- Outcome counts: `{json.dumps(metrics['outcome_class_counts'], sort_keys=True)}`.",
        f"- Recovery probability Brier: `{quality['binary_recovery_brier_score']:.6f}`; observed recovery-outcome frequency `{quality['binary_recovery_observed_rate']:.4f}`.",
        "- Confusion matrix rows are actual classes and columns predicted argmax classes; class metrics are in `metrics.json`.",
        "", "## Limitations", "",
        "The harm rate is illustrative; task-family IDs share one generator; only failure_prob varies among the four features; and simulator outcomes do not establish real-world agent behavior. Sensitivity runs reuse the primary-trained model and are stress tests, not model selection.",
    ])
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    policy_rows: list[dict[str, Any]],
    potential_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    harm_probability: float,
    model_checksum: str,
    preregistration_path: Path,
    training_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if training_rows is not None:
        _write_jsonl(output_dir / "training_examples.jsonl", training_rows)
    _write_jsonl(output_dir / "policy_runs.jsonl", policy_rows)
    _write_jsonl(output_dir / "evaluation_potential_outcomes.jsonl", potential_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison_summary.md").write_text(
        _summary_text(metrics, harm_probability, model_checksum), encoding="utf-8"
    )
    (output_dir / "scenario_design.md").write_text(SCENARIO_DESIGN, encoding="utf-8")
    (output_dir / "preregistration.md").write_text(
        preregistration_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    eval_keys = [
        (row["scenario_key"]["task_family"], row["scenario_key"]["root_seed"], row["scenario_key"]["seed"])
        for row in potential_rows
    ]
    actual_eval_families = sorted({key[0] for key in eval_keys})
    actual_eval_roots = sorted({key[1] for key in eval_keys})
    actual_train_families = sorted({row["scenario_key"]["task_family"] for row in training_rows}) if training_rows else None
    (output_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "mode": "synthetic_offline",
        "llm_calls": 0,
        "harm_probability": harm_probability,
        "model_sha256": model_checksum,
        "training_families": actual_train_families,
        "evaluation_families": actual_eval_families,
        "evaluation_root_seeds": actual_eval_roots,
        "scenario_count": len(potential_rows),
        "policy_run_count": len(policy_rows),
        "training_example_count": len(training_rows) if training_rows is not None else None,
        "replicate_keys_unique": len(eval_keys) == len(set(eval_keys)),
        "policy_names": list(POLICY_NAMES),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_primary(
    output_dir: str | Path,
    artifact_path: str | Path,
    preregistration_path: str | Path,
) -> dict[str, Any]:
    output_dir, artifact_path, prereg = Path(output_dir), Path(artifact_path), Path(preregistration_path)
    if not prereg.is_file():
        raise FileNotFoundError("frozen preregistration document must exist before held-out outcome generation")
    _ensure_fresh_output(output_dir, artifact_path)
    training_rows = generate_training_examples()
    model, metadata = _metadata_for_training(training_rows)
    scenarios = generate_evaluation_scenarios(PRIMARY_HARM_PROBABILITY)
    # Decisions are completely fixed before potential_outcomes is called for held-out scenarios.
    proposed = ProposedPolicy(model)
    provisional_checksum = "in-memory-training-pipeline"
    policy_rows, potential_rows = _run_policies_after_decisions(scenarios, proposed, provisional_checksum)
    _validate_record_sets(policy_rows, potential_rows, training_rows)
    observed_classes = {row["potential_outcomes"]["outcome_class"] for row in potential_rows}
    if observed_classes != set(CLASS_NAMES):
        missing = sorted(set(CLASS_NAMES) - observed_classes)
        raise RuntimeError(f"primary held-out scenarios did not contain all four outcome classes: {missing}")
    saved = save_model(model, metadata, artifact_path)
    checksum = hashlib.sha256(saved["artifact"].read_bytes()).hexdigest()
    for row in policy_rows:
        if row["policy"] == "proposed":
            row["model_sha256"] = checksum
    metrics = calculate_metrics(policy_rows, potential_rows)
    _write_outputs(
        output_dir, policy_rows, potential_rows, metrics,
        PRIMARY_HARM_PROBABILITY, checksum, prereg, training_rows,
    )
    return {
        "output_dir": output_dir,
        "artifact": saved["artifact"],
        "artifact_metadata": saved["metadata"],
        "training_example_count": len(training_rows),
        "scenario_count": len(scenarios),
        "policy_run_count": len(policy_rows),
        "metrics": metrics,
        "model_metadata": metadata,
    }


def run_sensitivity(
    output_dir: str | Path,
    artifact_path: str | Path,
    preregistration_path: str | Path,
    harm_probability: float,
) -> dict[str, Any]:
    if harm_probability not in SENSITIVITY_HARM_PROBABILITIES:
        raise ValueError("sensitivity harm probability is not preregistered")
    output_dir, artifact_path, prereg = Path(output_dir), Path(artifact_path), Path(preregistration_path)
    if not prereg.is_file():
        raise FileNotFoundError("frozen preregistration document is required")
    _ensure_fresh_output(output_dir)
    proposed, metadata, checksum = load_model(artifact_path)
    if metadata.get("training_harm_probability") != PRIMARY_HARM_PROBABILITY:
        raise ValueError("sensitivity must use the frozen primary-trained model")
    scenarios = generate_evaluation_scenarios(harm_probability)
    policy_rows, potential_rows = _run_policies_after_decisions(scenarios, proposed, checksum)
    _validate_record_sets(policy_rows, potential_rows)
    metrics = calculate_metrics(policy_rows, potential_rows)
    _write_outputs(output_dir, policy_rows, potential_rows, metrics, harm_probability, checksum, prereg)
    return {"output_dir": output_dir, "scenario_count": len(scenarios), "policy_run_count": len(policy_rows), "metrics": metrics}


def _json_equal(actual: Any, expected: Any) -> bool:
    return _metric_equal(actual, expected)


def audit_results(output_dir: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    """Recompute an experiment audit from JSON/metadata without deserializing a model."""
    output_dir, artifact_path = Path(output_dir), Path(artifact_path)
    manifest_path = output_dir / "run_manifest.json"
    policy_path = output_dir / "policy_runs.jsonl"
    potential_path = output_dir / "evaluation_potential_outcomes.jsonl"
    metrics_path = output_dir / "metrics.json"
    metadata_path = Path(str(artifact_path) + ".metadata.json")
    for path in (manifest_path, policy_path, potential_path, metrics_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Phase 9 audit file missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if checksum != sidecar.get("artifact_sha256") or checksum != manifest.get("model_sha256"):
        raise ValueError("model artifact checksum does not match recorded metadata")
    metadata = sidecar.get("model_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("model metadata sidecar is invalid")
    _validate_metadata(metadata)
    policy_rows, potential_rows = _read_jsonl(policy_path), _read_jsonl(potential_path)
    training_rows = _read_jsonl(output_dir / "training_examples.jsonl") if (output_dir / "training_examples.jsonl").exists() else None
    _validate_record_sets(policy_rows, potential_rows, training_rows)
    if any(row.get("model_sha256") != checksum for row in policy_rows if row["policy"] == "proposed"):
        raise ValueError("proposed-policy records do not identify the recorded model artifact")
    actual_eval_families = sorted({row["scenario_key"]["task_family"] for row in potential_rows})
    actual_eval_roots = sorted({row["scenario_key"]["root_seed"] for row in potential_rows})
    if (manifest.get("scenario_count") != len(potential_rows)
            or manifest.get("policy_run_count") != len(policy_rows)
            or manifest.get("evaluation_families") != actual_eval_families
            or manifest.get("evaluation_root_seeds") != actual_eval_roots
            or manifest.get("replicate_keys_unique") is not True
            or manifest.get("policy_names") != list(POLICY_NAMES)):
        raise ValueError("run manifest does not match the recorded evaluation rows")
    if training_rows is not None:
        actual_train_families = sorted({row["scenario_key"]["task_family"] for row in training_rows})
        if (manifest.get("training_families") != actual_train_families
                or manifest.get("training_example_count") != len(training_rows)
                or set(actual_train_families) & set(actual_eval_families)):
            raise ValueError("run manifest training/evaluation split does not match recorded rows")
    recomputed = calculate_metrics(policy_rows, potential_rows)
    recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not _json_equal(recomputed, recorded):
        raise ValueError("metrics do not reproduce from recorded JSONL")
    return {
        "scenario_count": recomputed["scenario_count"],
        "policy_run_count": recomputed["policy_run_count"],
        "training_example_count": len(training_rows) if training_rows is not None else None,
        "matched_scenarios": True,
        "train_evaluation_split": True,
        "feature_allowlist": True,
        "selected_outcomes_match": True,
        "metrics_match_jsonl": True,
        "artifact_checksum_matches": True,
        "model_deserialized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "sensitivity", "audit"), default="primary")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase9_cost_sensitive"))
    parser.add_argument("--artifact", type=Path, default=Path("models/phase9_cost_sensitive/proposed_policy.joblib"))
    parser.add_argument("--preregistration", type=Path, default=Path("results/phase9_cost_sensitive_preregistration.md"))
    parser.add_argument("--harm-probability", type=float, choices=SENSITIVITY_HARM_PROBABILITIES)
    args = parser.parse_args()
    if args.mode == "primary":
        result = run_primary(args.output_dir, args.artifact, args.preregistration)
        print(f"Phase 9 primary: synthetic_offline; scenarios={result['scenario_count']} runs={result['policy_run_count']} training={result['training_example_count']}")
        for policy, row in result["metrics"]["policies"].items():
            print(f"{policy}: success={row['successes']}/{row['attempts']} mean_cost={row['mean_realized_cost']:.4f} interventions={row['intervention_count']}")
    elif args.mode == "sensitivity":
        if args.harm_probability is None:
            parser.error("--mode sensitivity requires --harm-probability")
        result = run_sensitivity(args.output_dir, args.artifact, args.preregistration, args.harm_probability)
        print(f"Phase 9 sensitivity: harm_probability={args.harm_probability:.2f}; scenarios={result['scenario_count']} runs={result['policy_run_count']}")
    else:
        result = audit_results(args.output_dir, args.artifact)
        print(f"Phase 9 audit: PASS (read-only, model not deserialized; scenarios={result['scenario_count']} runs={result['policy_run_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
