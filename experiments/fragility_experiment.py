"""Leakage-controlled synthetic experiment for pre-action risk features.

This experiment is deliberately separate from the arithmetic benchmark and the
live Gemini path. It tests a policy mechanism under a declared stochastic
simulator; it does not model or validate real LLM-agent performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_demo import FEATURE_NAMES, NaiveThresholdPolicy, NoInterventionPolicy


EXPERIMENT_VERSION = "phase8.3-fragility-v1"
TRAIN_TASK_IDS = tuple(f"synthetic_task_{index:02d}" for index in range(12))
EVALUATION_TASK_IDS = tuple(f"synthetic_task_{index:02d}" for index in range(12, 16))
POLICIES = ("no_intervention", "naive_threshold", "proposed")

# This is the already-approved Phase 4 objective. "Helpful" means the offline
# continuation fails and the offline intervention branch passes.
COST_MATRIX = {
    "continue": {"not_helpful": 0.0, "helpful": 3.0},
    "intervene": {"not_helpful": 2.0, "helpful": 1.0},
}

# Predeclared causal simulator parameters. Fault prevalence is independent of
# latent execution fragility. A noisy sensor estimates baseline execution risk;
# it does not observe the fault treatment or any oracle outcome.
FAULT_PROBABILITY = 0.18
SENSOR_NOISE_SD = 0.18
FAULT_REPAIR_PROBABILITY = 0.65
BASE_FAILURE_INTERCEPT = 0.04
BASE_FAILURE_SLOPE = 0.76
INTERVENTION_FAILURE_INTERCEPT = 0.02
INTERVENTION_FAILURE_SLOPE = 0.35
REPAIR_OVERHEAD = 2.0
TASK_FAILURE_COST = 3.0
DEFAULT_ROOT_SEED = 20261028
DEFAULT_TRAIN_REPLICATES = 128
DEFAULT_EVALUATION_REPLICATES = 128

STREAM_NAMES = ("fragility", "sensor", "fault", "execution", "fault_repair")


@dataclass(frozen=True)
class SyntheticTask:
    """A task identity used only to group independent simulator scenarios."""

    task_id: str


@dataclass(frozen=True)
class SyntheticScenario:
    task_id: str
    seed: int
    root_seed: int
    latent_fragility: float
    noisy_fragility_signal: float
    observable_failure_prob: float
    fault_enabled: bool
    execution_uniform: float
    fault_repair_uniform: float
    scenario_hash: str
    initial_state_hash: str


TASKS = tuple(SyntheticTask(task_id) for task_id in (*TRAIN_TASK_IDS, *EVALUATION_TASK_IDS))
TASK_BY_ID = {task.task_id: task for task in TASKS}


class InvalidHelpfulProbability(ValueError):
    """A classifier output that cannot safely drive a policy decision."""


def calculate_decision_diagnostics(p_helpful: Real) -> dict[str, float | str]:
    """Calculate expected action costs and selection from helpfulness probability.

    This probability is the classifier's estimate that intervention will help;
    it is distinct from the `failure_prob` input feature.
    """
    if isinstance(p_helpful, bool) or not isinstance(p_helpful, Real):
        raise InvalidHelpfulProbability("classifier p_helpful must be a numeric probability")
    probability = float(p_helpful)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise InvalidHelpfulProbability("classifier p_helpful must be finite and within [0, 1]")
    continue_cost = 3.0 * probability
    intervene_cost = 2.0 - probability
    return {
        "p_helpful": probability,
        "continue_expected_cost": continue_cost,
        "intervene_expected_cost": intervene_cost,
        "cost_difference": intervene_cost - continue_cost,
        "selected_action": "intervene" if probability > 0.5 else "continue",
    }


def _stream_seed(root_seed: int, task_id: str, seed: int, stream_name: str) -> int:
    """Derive a stable, domain-separated PRNG seed for one scenario property."""
    if stream_name not in STREAM_NAMES:
        raise ValueError(f"unknown random stream: {stream_name}")
    material = json.dumps(
        [EXPERIMENT_VERSION, int(root_seed), task_id, int(seed), stream_name],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _rng(root_seed: int, task_id: str, seed: int, stream_name: str) -> random.Random:
    return random.Random(_stream_seed(root_seed, task_id, seed, stream_name))


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def features_from_observation(observable_failure_prob: float) -> dict[str, float]:
    """The single training/inference feature path; accepts no hidden metadata."""
    if isinstance(observable_failure_prob, bool) or not isinstance(observable_failure_prob, (int, float)):
        raise ValueError("observable failure probability must be numeric")
    if not math.isfinite(observable_failure_prob) or not 0 <= observable_failure_prob <= 1:
        raise ValueError("observable failure probability must be finite and in [0, 1]")
    features = {
        "failure_prob": float(observable_failure_prob),
        "reversibility_proxy": 0.25,
        "trajectory_position": 0.0,
        "disruption_history": 0.0,
    }
    if tuple(features) != FEATURE_NAMES:
        raise AssertionError("synthetic experiment feature order diverged from the project schema")
    return features


def generate_scenario(task: SyntheticTask, seed: int, root_seed: int = DEFAULT_ROOT_SEED) -> SyntheticScenario:
    """Generate hidden causes, an observable noisy estimate, and action noise.

    Latent fragility is an iid Uniform(0,1) draw in its own keyed stream. Its
    value affects execution failure probability but is not passed to a policy.
    A separate sensor stream adds Gaussian noise to the observable estimate.
    Fault treatment, execution noise, and repair success use three other
    domain-separated streams. Task IDs key reproducibility only; they do not
    change any variable's distribution and are never policy features.
    """
    latent = _rng(root_seed, task.task_id, seed, "fragility").random()
    sensor = _rng(root_seed, task.task_id, seed, "sensor")
    observed_fragility = _clip_probability(sensor.gauss(latent, SENSOR_NOISE_SD))
    baseline_execution_risk = BASE_FAILURE_INTERCEPT + BASE_FAILURE_SLOPE * observed_fragility
    # Fault status is unknown to the policy. Its independent prior is included
    # in the risk estimate, but the realized fault draw is not.
    observable_failure_prob = 1.0 - (1.0 - baseline_execution_risk) * (1.0 - FAULT_PROBABILITY)
    fault_enabled = _rng(root_seed, task.task_id, seed, "fault").random() < FAULT_PROBABILITY
    execution_uniform = _rng(root_seed, task.task_id, seed, "execution").random()
    repair_uniform = _rng(root_seed, task.task_id, seed, "fault_repair").random()
    empty_state = {"completed_actions": []}
    initial_state_hash = hashlib.sha256(json.dumps(empty_state, sort_keys=True).encode()).hexdigest()
    payload = {
        "experiment": EXPERIMENT_VERSION,
        "task_id": task.task_id,
        "seed": int(seed),
        "root_seed": int(root_seed),
        "latent_fragility": latent,
        "noisy_fragility_signal": observed_fragility,
        "observable_failure_prob": observable_failure_prob,
        "fault_enabled": fault_enabled,
        "execution_uniform": execution_uniform,
        "fault_repair_uniform": repair_uniform,
    }
    scenario_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return SyntheticScenario(
        task_id=task.task_id,
        seed=int(seed),
        root_seed=int(root_seed),
        latent_fragility=latent,
        noisy_fragility_signal=observed_fragility,
        observable_failure_prob=observable_failure_prob,
        fault_enabled=fault_enabled,
        execution_uniform=execution_uniform,
        fault_repair_uniform=repair_uniform,
        scenario_hash=scenario_hash,
        initial_state_hash=initial_state_hash,
    )


def _branch_outcome(scenario: SyntheticScenario, action: str) -> bool:
    """Return verifier-like success for one action; called at execution time."""
    if action not in {"continue", "intervene"}:
        raise ValueError("action must be continue or intervene")
    baseline_risk = BASE_FAILURE_INTERCEPT + BASE_FAILURE_SLOPE * scenario.latent_fragility
    if action == "continue":
        return not scenario.fault_enabled and scenario.execution_uniform >= baseline_risk
    intervention_risk = (
        INTERVENTION_FAILURE_INTERCEPT
        + INTERVENTION_FAILURE_SLOPE * scenario.latent_fragility
    )
    fault_repaired = (
        not scenario.fault_enabled
        or scenario.fault_repair_uniform < FAULT_REPAIR_PROBABILITY
    )
    return fault_repaired and scenario.execution_uniform >= intervention_risk


def simulate_counterfactual_branches(scenario: SyntheticScenario) -> dict[str, bool]:
    """Offline-only potential outcomes for training labels; never used at inference."""
    return {
        "continue_passed": _branch_outcome(scenario, "continue"),
        "intervene_passed": _branch_outcome(scenario, "intervene"),
    }


def _cost_pair(continue_passed: bool, intervene_passed: bool) -> tuple[float, float, int]:
    helpful = int(intervene_passed and not continue_passed)
    class_name = "helpful" if helpful else "not_helpful"
    return (
        COST_MATRIX["continue"][class_name],
        COST_MATRIX["intervene"][class_name],
        helpful,
    )


def make_training_example(task: SyntheticTask, seed: int, root_seed: int = DEFAULT_ROOT_SEED) -> dict[str, Any]:
    scenario = generate_scenario(task, seed, root_seed)
    branches = simulate_counterfactual_branches(scenario)
    continue_cost, intervene_cost, helpful = _cost_pair(**branches)
    return {
        "task_id": task.task_id,
        "seed": int(seed),
        "scenario_hash": scenario.scenario_hash,
        "features": features_from_observation(scenario.observable_failure_prob),
        "feature_names": list(FEATURE_NAMES),
        "label_intervene_helped": helpful,
        "continue_cost": continue_cost,
        "intervene_cost": intervene_cost,
        "branches": branches,
        # Ground truth is kept in metadata for audit and excluded from features.
        "scenario_metadata": {
            "latent_fragility": scenario.latent_fragility,
            "noisy_fragility_signal": scenario.noisy_fragility_signal,
            "fault_enabled": scenario.fault_enabled,
        },
    }


def generate_training_examples(
    seeds: Iterable[int] = range(DEFAULT_TRAIN_REPLICATES),
    root_seed: int = DEFAULT_ROOT_SEED,
) -> list[dict[str, Any]]:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be non-empty and unique")
    return [make_training_example(TASK_BY_ID[task_id], seed, root_seed)
            for task_id in TRAIN_TASK_IDS for seed in seeds]


class FragilityProposedPolicy:
    """Experiment-local policy: one probability prediction and fixed cost choice."""

    def __init__(self, model: Pipeline):
        if not hasattr(model, "predict_proba") or set(model.classes_.tolist()) != {0, 1}:
            raise ValueError("fragility artifact must contain a fitted binary probability model")
        self.model = model

    def decide(self, features: dict[str, float]) -> dict[str, float | str]:
        if tuple(features) != FEATURE_NAMES:
            raise ValueError("features must use the project's exact ordered schema")
        row = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
        if row.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(row).all() or ((row < 0) | (row > 1)).any():
            raise ValueError("features must be one finite four-value row in [0, 1]")
        probabilities = np.asarray(self.model.predict_proba(row))  # exactly one classifier prediction
        if probabilities.shape != (1, len(self.model.classes_)):
            raise InvalidHelpfulProbability("classifier returned a malformed probability matrix")
        diagnostics = calculate_decision_diagnostics(
            probabilities[0, list(self.model.classes_).index(1)]
        )
        helpful_probability = diagnostics["p_helpful"]
        return {
            "decision": diagnostics["selected_action"],
            "helpful_intervention_probability": helpful_probability,
            "expected_continue_cost": diagnostics["continue_expected_cost"],
            "expected_intervene_cost": diagnostics["intervene_expected_cost"],
            "decision_diagnostics": diagnostics,
        }


def train_model(examples: list[dict[str, Any]]) -> tuple[Pipeline, dict[str, Any]]:
    if not examples:
        raise ValueError("training examples cannot be empty")
    if any(item["task_id"] not in TRAIN_TASK_IDS for item in examples):
        raise ValueError("held-out task identities cannot enter model training")
    labels = np.asarray([item["label_intervene_helped"] for item in examples], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("training data must contain both helpfulness classes")
    features = np.asarray(
        [[item["features"][name] for name in FEATURE_NAMES] for item in examples], dtype=float
    )
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(random_state=0, solver="liblinear", max_iter=1000)),
    ])
    model.fit(features, labels)
    return model, {
        "experiment_version": EXPERIMENT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "risk_signal": "noisy_pre_action_fragility_estimate_with_independent_fault_prior",
        "training_tasks": list(TRAIN_TASK_IDS),
        "evaluation_tasks": list(EVALUATION_TASK_IDS),
        "split_unit": "synthetic_task_id",
        "cost_matrix": COST_MATRIX,
        "scenario_generation": {
            "root_seed": DEFAULT_ROOT_SEED,
            "fault_probability": FAULT_PROBABILITY,
            "sensor_noise_sd": SENSOR_NOISE_SD,
            "fault_repair_probability": FAULT_REPAIR_PROBABILITY,
            "baseline_failure_probability": f"{BASE_FAILURE_INTERCEPT} + {BASE_FAILURE_SLOPE} * latent_fragility",
            "intervention_failure_probability": f"{INTERVENTION_FAILURE_INTERCEPT} + {INTERVENTION_FAILURE_SLOPE} * latent_fragility",
            "independent_streams": list(STREAM_NAMES),
        },
        "limitations": [
            "Synthetic mechanism only; it does not validate real LLM-agent performance.",
            "Task identities are iid strata from one generator, not distinct real-world domains.",
            "Noisy risk estimates exclude the realized independent injected fault.",
        ],
    }


def save_model(model: Pipeline, metadata: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)
    return path


def load_model(path: str | Path) -> FragilityProposedPolicy:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError("invalid fragility experiment artifact")
    metadata = artifact["metadata"]
    if (metadata.get("experiment_version") != EXPERIMENT_VERSION
            or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("cost_matrix") != COST_MATRIX
            or metadata.get("risk_signal") != "noisy_pre_action_fragility_estimate_with_independent_fault_prior"):
        raise ValueError("fragility experiment artifact metadata is incompatible")
    if set(metadata.get("training_tasks", ())) & set(metadata.get("evaluation_tasks", ())):
        raise ValueError("artifact task split overlaps")
    return FragilityProposedPolicy(artifact.get("model"))


def _run_policy(scenario: SyntheticScenario, policy_name: str, proposed: FragilityProposedPolicy | None) -> dict[str, Any]:
    features = features_from_observation(scenario.observable_failure_prob)
    # Each policy gets its own fresh state object even though this small
    # simulator's action state is deliberately empty at the decision boundary.
    initial_state = {"completed_actions": []}
    initial_state_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
    if initial_state_hash != scenario.initial_state_hash:
        raise AssertionError("scenario and execution initial-state definitions differ")
    if policy_name == "no_intervention":
        decision = NoInterventionPolicy().decide(features["failure_prob"], features)
        decision_details = None
    elif policy_name == "naive_threshold":
        decision = NaiveThresholdPolicy().decide(features["failure_prob"], features)
        decision_details = None
    elif policy_name == "proposed":
        if proposed is None:
            raise ValueError("proposed policy requires the saved fragility artifact")
        decision_details = proposed.decide(features)
        decision = decision_details["decision"]
    else:
        raise ValueError(f"unknown policy {policy_name}")

    if decision == "abort":
        passed = False  # An abort has no verifier pass and remains a failed task.
        executed_steps = 0
        intervention = False
    elif decision == "continue":
        passed = _branch_outcome(scenario, "continue")
        executed_steps = 1
        intervention = False
    elif decision == "intervene":
        passed = _branch_outcome(scenario, "intervene")
        executed_steps = 2  # corrective guidance/replan plus the action
        intervention = True
    else:
        raise ValueError(f"unsupported decision {decision}")

    return {
        "policy": policy_name,
        "task_id": scenario.task_id,
        "seed": scenario.seed,
        "scenario_hash": scenario.scenario_hash,
        "initial_state": initial_state,
        "initial_state_hash": initial_state_hash,
        "features": features,
        "feature_names": list(FEATURE_NAMES),
        "decision": decision,
        # Additive and optional for older readers; these are predictions, not
        # realized costs, which are computed later by _realized_cost/_metrics.
        "decision_diagnostics": (
            decision_details.get("decision_diagnostics")
            if isinstance(decision_details, dict) else None
        ),
        "intervention": intervention,
        "corrective_guidance": (
            "Replan the current action using the observable fragility estimate and preserve the current state."
            if intervention else None
        ),
        "planner_replanned": intervention,
        "passed": bool(passed),
        "aborted": decision == "abort",
        "steps_executed": executed_steps,
        "local_latency_seconds": None,
    }


def _realized_cost(run: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Score observed runs using the fixed objective, without evaluation branching.

    The approved matrix gives costs for both-pass and successful-recovery pairs.
    For an intervention that fails, the experiment extends the not-helpful cost
    by the declared task-failure penalty. Continue/abort failures cost 3. A
    successful intervention costs 1 when the matched control failed and 2 when
    the matched control passed. No policy receives the matched outcome.
    """
    if not run["intervention"]:
        return TASK_FAILURE_COST if not run["passed"] else 0.0
    if not run["passed"]:
        return COST_MATRIX["intervene"]["not_helpful"] + TASK_FAILURE_COST
    return COST_MATRIX["intervene"]["helpful"] if not baseline["passed"] else COST_MATRIX["intervene"]["not_helpful"]


def _reliability_summary(probabilities: list[float], outcomes: list[bool], bins: int = 10) -> dict[str, Any]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("reliability summary requires paired non-empty observations")
    brier = sum((p - float(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)
    rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(p, y) for p, y in zip(probabilities, outcomes)
                   if low <= p < high or (index == bins - 1 and p == 1.0)]
        if members:
            rows.append({
                "bin_low": low,
                "bin_high": high,
                "count": len(members),
                "mean_predicted": sum(p for p, _ in members) / len(members),
                "observed_failure_rate": sum(bool(y) for _, y in members) / len(members),
            })
    return {"brier_score": brier, "reliability_bins": rows}


def _metrics(scenarios: list[SyntheticScenario], runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["task_id"], row["seed"], row["policy"]): row for row in runs}
    keys = [(scenario.task_id, scenario.seed) for scenario in scenarios]
    if len(by_key) != len(scenarios) * len(POLICIES):
        raise ValueError("evaluation must have exactly one run per policy and scenario")
    baseline_success = {key: by_key[(*key, "no_intervention")]["passed"] for key in keys}
    output: dict[str, Any] = {}
    for policy in POLICIES:
        policy_runs = [by_key[(*key, policy)] for key in keys]
        successes = sum(row["passed"] for row in policy_runs)
        interventions = [row for row in policy_runs if row["intervention"]]
        control_success_keys = [key for key in keys if baseline_success[key]]
        control_failure_keys = [key for key in keys if not baseline_success[key]]
        disruption_count = sum(by_key[(*key, policy)]["intervention"] for key in control_success_keys)
        recoveries = sum(
            by_key[(*key, policy)]["intervention"] and by_key[(*key, policy)]["passed"]
            for key in control_failure_keys
        )
        unnecessary = disruption_count
        costs = [_realized_cost(by_key[(*key, policy)], by_key[(*key, "no_intervention")]) for key in keys]
        output[policy] = {
            "attempts": len(policy_runs),
            "successes": successes,
            "task_success_rate": successes / len(policy_runs),
            "intervention_count": len(interventions),
            "intervention_rate": len(interventions) / len(policy_runs),
            "abort_count": sum(row["aborted"] for row in policy_runs),
            "disruption_count": disruption_count,
            "disruption_denominator": len(control_success_keys),
            "disruption_rate": disruption_count / len(control_success_keys) if control_success_keys else None,
            "unnecessary_intervention_count": unnecessary,
            "unnecessary_intervention_denominator": len(control_success_keys),
            "unnecessary_intervention_rate": unnecessary / len(control_success_keys) if control_success_keys else None,
            "recovery_count": recoveries,
            "recovery_denominator": len(control_failure_keys),
            "recovery_rate": recoveries / len(control_failure_keys) if control_failure_keys else None,
            "mean_realized_cost": sum(costs) / len(costs),
            "average_steps": sum(row["steps_executed"] for row in policy_runs) / len(policy_runs),
            "average_local_runtime_seconds": sum(row["local_latency_seconds"] for row in policy_runs) / len(policy_runs),
        }
    baseline_rows = [by_key[(*key, "no_intervention")] for key in keys]
    output["risk_calibration"] = _reliability_summary(
        [row["features"]["failure_prob"] for row in baseline_rows],
        [not row["passed"] for row in baseline_rows],
    )
    return output


def validate_matched_runs(runs: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in runs:
        key = (row["task_id"], row["seed"])
        policies = grouped.setdefault(key, {})
        if row["policy"] in policies:
            raise ValueError(f"duplicate policy result for {key}")
        policies[row["policy"]] = row
    if not grouped:
        raise ValueError("evaluation has no runs")
    for key, policies in grouped.items():
        if set(policies) != set(POLICIES):
            raise ValueError(f"scenario {key} does not have exactly one result per policy")
        reference = policies[POLICIES[0]]
        for row in policies.values():
            if (row["scenario_hash"], row["initial_state_hash"], row["features"], row["initial_state"],
                    row.get("scenario_metadata")) != (
                reference["scenario_hash"], reference["initial_state_hash"], reference["features"],
                reference["initial_state"], reference.get("scenario_metadata")
            ):
                raise ValueError(f"scenario {key} is not matched across policies")


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def run_experiment(
    output_dir: str | Path,
    artifact_path: str | Path,
    root_seed: int = DEFAULT_ROOT_SEED,
    train_replicates: int = DEFAULT_TRAIN_REPLICATES,
    evaluation_replicates: int = DEFAULT_EVALUATION_REPLICATES,
) -> dict[str, Any]:
    if train_replicates <= 0 or evaluation_replicates <= 0:
        raise ValueError("replicate counts must be positive")
    output_dir, artifact_path = Path(output_dir), Path(artifact_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing experiment outputs: {output_dir}")
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment model: {artifact_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = generate_training_examples(range(train_replicates), root_seed)
    model, metadata = train_model(training)
    metadata["scenario_generation"]["root_seed"] = root_seed
    metadata["training_replicates_per_task"] = train_replicates
    metadata["evaluation_replicates_per_task"] = evaluation_replicates
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact_path)
    train_path = output_dir / "training_examples.jsonl"
    _jsonl_write(train_path, training)

    proposed = load_model(artifact_path)
    scenarios = [generate_scenario(TASK_BY_ID[task_id], seed, root_seed)
                 for task_id in EVALUATION_TASK_IDS for seed in range(evaluation_replicates)]
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for policy_name in POLICIES:
            started = time.perf_counter()
            row = _run_policy(scenario, policy_name, proposed if policy_name == "proposed" else None)
            row["local_latency_seconds"] = time.perf_counter() - started
            row["experiment"] = EXPERIMENT_VERSION
            row["scenario_metadata"] = {
                "fault_enabled": scenario.fault_enabled,
                "latent_fragility": scenario.latent_fragility,
                "noisy_fragility_signal": scenario.noisy_fragility_signal,
            }
            row["llm_calls"] = 0
            row["mode"] = "synthetic_offline"
            runs.append(row)
    validate_matched_runs(runs)
    metrics = _metrics(scenarios, runs)
    runs_path = output_dir / "evaluation_runs.jsonl"
    _jsonl_write(runs_path, runs)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_dir / "comparison_summary.md"
    _write_summary(summary_path, metrics, len(scenarios), root_seed, train_replicates, evaluation_replicates)
    design_path = output_dir / "scenario_design.md"
    _write_design(design_path, root_seed, train_replicates, evaluation_replicates)
    return {
        "experiment": EXPERIMENT_VERSION,
        "scenario_count": len(scenarios),
        "run_count": len(runs),
        "training_example_count": len(training),
        "training_examples": train_path,
        "evaluation_runs": runs_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "design": design_path,
        "artifact": artifact_path,
        "measured_metrics": metrics,
    }


def audit_saved_results(output_dir: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    """Read-only validation and metric recomputation from recorded experiment files."""
    output_dir, artifact_path = Path(output_dir), Path(artifact_path)
    runs_path = output_dir / "evaluation_runs.jsonl"
    metrics_path = output_dir / "metrics.json"
    training_path = output_dir / "training_examples.jsonl"
    for path in (runs_path, metrics_path, training_path, artifact_path):
        if not path.is_file():
            raise FileNotFoundError(f"required experiment artifact missing: {path}")

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            rows.append(row)
        if not rows:
            raise ValueError(f"empty experiment JSONL: {path}")
        return rows

    runs = read_jsonl(runs_path)
    training = read_jsonl(training_path)
    validate_matched_runs(runs)
    for row in runs:
        if row.get("mode") != "synthetic_offline" or row.get("llm_calls") != 0:
            raise ValueError("evaluation log contains a non-offline run")
        if row.get("feature_names") != list(FEATURE_NAMES) or set(row.get("features", {})) != set(FEATURE_NAMES):
            raise ValueError("evaluation feature schema/order does not match")
        forbidden = {"fault_enabled", "latent_fragility", "scenario_hash", "seed", "task_id",
                     "passed", "expected", "actual", "branches"}
        if forbidden & set(row["features"]):
            raise ValueError("evaluation policy features contain ground-truth metadata")
    if any(row.get("task_id") not in TRAIN_TASK_IDS for row in training):
        raise ValueError("training log contains a held-out evaluation task")
    artifact = joblib.load(artifact_path)
    metadata = artifact.get("metadata") if isinstance(artifact, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("saved model artifact has invalid metadata")
    if (set(metadata.get("training_tasks", ())) != set(TRAIN_TASK_IDS)
            or set(metadata.get("evaluation_tasks", ())) != set(EVALUATION_TASK_IDS)
            or set(metadata["training_tasks"]) & set(metadata["evaluation_tasks"])):
        raise ValueError("saved artifact train/evaluation task split is invalid")

    grouped = {}
    for row in runs:
        grouped.setdefault((row["task_id"], row["seed"]), row)
    scenario_rows = [SimpleNamespace(task_id=task_id, seed=seed)
                     for task_id, seed in grouped]
    recomputed = _metrics(scenario_rows, runs)
    recorded = json.loads(metrics_path.read_text(encoding="utf-8"))

    def equal(actual: Any, expected: Any) -> bool:
        if isinstance(actual, dict) and isinstance(expected, dict):
            return set(actual) == set(expected) and all(equal(actual[key], expected[key]) for key in actual)
        if isinstance(actual, list) and isinstance(expected, list):
            return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return isinstance(expected, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        return actual == expected

    if not equal(recomputed, recorded):
        raise ValueError("recorded metrics do not match direct JSONL recomputation")
    return {
        "scenario_count": len(scenario_rows),
        "run_count": len(runs),
        "training_example_count": len(training),
        "matched_scenarios": True,
        "offline_only": True,
        "metrics_match_jsonl": True,
        "metrics": recomputed,
    }


def _write_summary(path: Path, metrics: dict[str, Any], scenario_count: int,
                   root_seed: int, train_replicates: int, evaluation_replicates: int) -> None:
    lines = [
        "# Phase 8.3 synthetic fragility experiment",
        "",
        "This is an offline synthetic experiment. It tests the decision-policy mechanism under the declared simulator; it does not establish real-world LLM-agent performance.",
        "",
        f"- Root seed: `{root_seed}`; training replicates per task: `{train_replicates}`; held-out replicates per task: `{evaluation_replicates}`.",
        f"- Held-out scenarios: `{scenario_count}` across tasks `{', '.join(EVALUATION_TASK_IDS)}`; model training used only `{', '.join(TRAIN_TASK_IDS)}`.",
        "- Primary metric: mean realized cost. Continue/abort costs are 0 on verifier pass and 3 on failure. Intervention costs are 1 for a verifier-passing recovery, 2 for a verifier-passing unnecessary intervention, and 5 for a failed intervention (the fixed not-helpful intervention cost 2 plus failure penalty 3).",
        "- Disruption and unnecessary-intervention denominators are matched scenarios where the no-intervention control passed. Recovery denominator is matched scenarios where that control failed; recovery requires an intervention and a verifier pass.",
        "- Average steps counts 0 for abort, 1 for continue, and 2 for corrective replan plus action. Runtime is local simulator wall time, not provider latency.",
        "- Risk Brier score uses each held-out no-intervention verifier outcome as the failure label. Reliability bins are descriptive; no significance claim is made.",
        "- Fault status is generated independently from fragility, sensor noise, execution noise, and repair success via domain-separated PRNG streams. Only noisy estimated risk and fixed action metadata enter features; hidden scenario metadata is retained separately for audit.",
        "",
        "| Policy | Mean cost | Success | Interventions | Abort | Disruption | Recovery | Avg steps | Local runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = metrics[policy]
        disruption = "n/a" if row["disruption_rate"] is None else f"{row['disruption_count']}/{row['disruption_denominator']} ({row['disruption_rate']:.1%})"
        recovery = "n/a" if row["recovery_rate"] is None else f"{row['recovery_count']}/{row['recovery_denominator']} ({row['recovery_rate']:.1%})"
        lines.append(
            f"| {policy} | {row['mean_realized_cost']:.4f} | {row['successes']}/{row['attempts']} ({row['task_success_rate']:.1%}) | "
            f"{row['intervention_count']}/{row['attempts']} ({row['intervention_rate']:.1%}) | {row['abort_count']} | "
            f"{disruption} | {recovery} | {row['average_steps']:.3f} | {row['average_local_runtime_seconds']:.6f} |"
        )
    calibration = metrics["risk_calibration"]
    lines.extend(["", f"Risk estimate Brier score: `{calibration['brier_score']:.6f}`.", "", "No policy outcome is tuned against this held-out set."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_design(path: Path, root_seed: int, train_replicates: int,
                  evaluation_replicates: int) -> None:
    text = f"""# Phase 8.3 scenario-generation and evaluation design

This is a **synthetic offline mechanism experiment**, not a live LLM experiment
or evidence about real-world multi-agent performance.

## Predeclared causal simulator

For each (task identity, replicate seed), the generator uses separately
domain-separated SHA-256-derived Python PRNG streams named {', '.join(STREAM_NAMES)}.
Task identities and row order are grouping keys only: no identifier, filename,
seed, hash, or row index is supplied to the policy. Reordering generation does
not change a keyed scenario.

1. Latent execution fragility `L ~ Uniform(0, 1)` is drawn from the `fragility`
   stream. It describes stochastic execution sensitivity, not a fault flag.
2. The observable sensor value is `clip(L + Normal(0, {SENSOR_NOISE_SD}), 0, 1)`
   from the independent `sensor` stream. Baseline execution risk is
   `{BASE_FAILURE_INTERCEPT} + {BASE_FAILURE_SLOPE} * sensor_value`. The visible
   `failure_prob` additionally applies the fixed independent-fault prior
   {FAULT_PROBABILITY}: `1 - (1 - baseline_risk) * (1 - {FAULT_PROBABILITY})`.
   The actual hidden fault draw is never used to calculate this feature.
3. Hidden injected fault `F ~ Bernoulli({FAULT_PROBABILITY})` is drawn from the
   independent `fault` stream. It is used in outcome generation and audit
   metadata only.
4. One exogenous execution draw `U ~ Uniform(0, 1)` is generated from the
   `execution` stream and shared by potential action outcomes. Continuing
   passes iff `not F and U >= {BASE_FAILURE_INTERCEPT} + {BASE_FAILURE_SLOPE} * L`.
5. Intervention lowers execution failure probability to
   `{INTERVENTION_FAILURE_INTERCEPT} + {INTERVENTION_FAILURE_SLOPE} * L` and can
   repair a hidden fault with probability {FAULT_REPAIR_PROBABILITY} using the
   independent `fault_repair` stream. This permits useful, unnecessary, and
   ineffective interventions. Because the simulator shares `U` and lowers the
   fragility threshold, intervention cannot turn a passing continuation into
   a failure; harmful state-corruption effects are outside this experiment.

## Features and labels

The sole extractor accepts only the scalar observable risk estimate. Ordered
features are `{', '.join(FEATURE_NAMES)}`. The final three values are fixed
one-step action metadata. Latent fragility, fault status, expected answers,
verifier outcomes, scenario hashes, seeds, IDs, filenames, order, and branch
outcomes are excluded. At inference, the proposed policy receives this feature
dictionary and makes one classifier prediction; it does not branch or inspect
counterfactual outcomes. Corrective guidance/replanning is represented as a
local synthetic action with two counted steps; no rollback occurs.

Training labels use matched offline continuation/intervention outcomes only:
`helpful = continuation_failed and intervention_passed`. The saved classifier
uses the existing Phase 4 cost matrix:

| Action | Not helpful | Helpful |
|---|---:|---:|
| continue | 0 | 3 |
| intervene | 2 | 1 |

Training uses tasks `{', '.join(TRAIN_TASK_IDS)}`; held-out evaluation uses
`{', '.join(EVALUATION_TASK_IDS)}`. Default root seed is `{DEFAULT_ROOT_SEED}`;
this run uses root seed `{root_seed}`, {train_replicates} replicates per training
task, and {evaluation_replicates} replicates per held-out task. No held-out
outcome is used for training, threshold choice, or cost tuning.

## Predeclared metrics

The primary metric is mean realized cost. For observed matched policy runs,
continue/abort costs are 0 on verifier pass and 3 on failure. An intervention
that passes costs 1 when the matched no-intervention control failed (recovery)
and 2 when it passed (unnecessary intervention). A failed intervention costs
5: the fixed not-helpful intervention cost 2 plus a failure penalty of 3. This
explicitly completes outcome cases not represented in the original Phase 4
binary helpfulness matrix. Evaluation uses the actually recorded baseline and
policy outcomes; it does not run extra counterfactual evaluation branches.

Task success is verifier passes / attempts. Intervention rate is intervening
runs / attempts. Disruption and unnecessary-intervention denominators are
matched cases where no-intervention passed; their numerators are interventions
on those cases. Recovery denominator is matched no-intervention failures;
numerator is intervention plus a verifier pass. Average steps counts 0 for
abort, 1 for continue, and 2 for corrective replan plus action. Runtime is local
simulation wall time. Brier score compares the risk estimate with the matched
no-intervention failure outcome once per scenario; reliability bins are
descriptive only. This small synthetic set supports no statistical-significance
claim.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase8_3_fragility"))
    parser.add_argument("--artifact", type=Path, default=Path("models/phase8_3_fragility/proposed_policy.joblib"))
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--train-replicates", type=int, default=DEFAULT_TRAIN_REPLICATES)
    parser.add_argument("--evaluation-replicates", type=int, default=DEFAULT_EVALUATION_REPLICATES)
    parser.add_argument("--audit", action="store_true", help="recompute and validate existing outputs without writing")
    args = parser.parse_args()
    if args.audit:
        audit = audit_saved_results(args.output_dir, args.artifact)
        print(f"Audit: PASS (read-only, offline; {audit['scenario_count']} scenarios, {audit['run_count']} runs, {audit['training_example_count']} training examples)")
        print("Matched scenarios, task split, feature allowlist, offline mode, and metrics from raw JSONL: PASS")
        return 0
    result = run_experiment(args.output_dir, args.artifact, args.root_seed,
                            args.train_replicates, args.evaluation_replicates)
    print(f"Experiment: {result['experiment']} (synthetic offline; llm_calls=0)")
    print(f"Scenarios: {result['scenario_count']} | policy runs: {result['run_count']} | training examples: {result['training_example_count']}")
    print(f"Summary: {result['summary']}")
    for policy in POLICIES:
        row = result["measured_metrics"][policy]
        print(f"{policy}: mean_cost={row['mean_realized_cost']:.4f} success={row['successes']}/{row['attempts']} interventions={row['intervention_count']} aborts={row['abort_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
