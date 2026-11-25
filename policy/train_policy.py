"""Generate offline counterfactual labels and train the proposed policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_demo import (
    FEATURE_NAMES,
    TASKS,
    Coder,
    Planner,
    Verifier,
    features_from_observation,
    make_scenario,
)
from policy.proposed import ARTIFACT_VERSION, RISK_SIGNAL_ID


COST_MATRIX = {
    "continue": {"not_helpful": 0.0, "helpful": 3.0},
    "intervene": {"not_helpful": 2.0, "helpful": 1.0},
}
TRAIN_TASK_IDS = tuple(task.task_id for task in TASKS[:6])
EVALUATION_TASK_IDS = tuple(task.task_id for task in TASKS[6:])
TRAINING_SEEDS = tuple(range(32))
ARTIFACT_PATH = Path("models/proposed_policy.joblib")
DATASET_PATH = Path("results/offline_training_examples.jsonl")


def _evaluate_instruction(task: Any, instruction: str) -> dict[str, Any]:
    code = Coder().implement(instruction, "")
    passed, evidence = Verifier().verify(code, task)
    return {"code": code, "verifier": evidence, "passed": passed}


def simulate_counterfactual_branch(task: Any, seed: int, action: str) -> dict[str, Any]:
    """Run one isolated offline continuation from the same empty prefix."""
    if action not in {"continue", "intervene"}:
        raise ValueError("offline branch action must be continue or intervene")
    scenario = make_scenario(task, seed)
    canonical_step = Planner().plan(task)[0]
    initial_state = {"code": "", "completed_steps": []}
    if action == "continue" and scenario.fault_enabled:
        instruction = f"Return the integer result {scenario.wrong_answer}."
    else:
        instruction = canonical_step
    result = _evaluate_instruction(task, instruction)
    return {
        "branch_type": "offline_counterfactual",
        "action": action,
        "task_id": task.task_id,
        "seed": seed,
        "scenario_hash": scenario.scenario_hash,
        "fault_enabled": scenario.fault_enabled,
        "initial_state": initial_state,
        "instruction": instruction,
        **result,
    }


def _cost_pair(continue_passed: bool, intervene_passed: bool) -> tuple[float, float]:
    """Return actual branch costs for the approved cost matrix's defined outcomes."""
    if continue_passed and intervene_passed:
        return 0.0, 2.0
    if not continue_passed and intervene_passed:
        return 3.0, 1.0
    raise ValueError("the approved cost matrix does not define this branch outcome pair")


def make_example(task: Any, seed: int) -> dict[str, Any]:
    continue_branch = simulate_counterfactual_branch(task, seed, "continue")
    intervene_branch = simulate_counterfactual_branch(task, seed, "intervene")
    if (continue_branch["scenario_hash"] != intervene_branch["scenario_hash"]
            or continue_branch["initial_state"] != intervene_branch["initial_state"]):
        raise AssertionError("counterfactual branches must share scenario and initial state")
    continue_cost, intervene_cost = _cost_pair(
        continue_branch["passed"], intervene_branch["passed"]
    )
    scenario = make_scenario(task, seed)
    features = features_from_observation(
        continue_branch["instruction"], "", "code_assignment", 0, 1,
        prior_interventions=0,
    )
    prefix_payload = {"scenario_hash": scenario.scenario_hash, "step_index": 0, "initial_state": {"code": "", "completed_steps": []}}
    prefix_id = hashlib.sha256(json.dumps(prefix_payload, sort_keys=True).encode()).hexdigest()
    intervene_helped = int(intervene_branch["passed"] and not continue_branch["passed"])
    return {
        "task_id": task.task_id,
        "seed": seed,
        "scenario_hash": scenario.scenario_hash,
        "prefix_id": prefix_id,
        "features": features,
        "feature_names": list(FEATURE_NAMES),
        "continue_branch": continue_branch,
        "intervene_branch": intervene_branch,
        "continue_cost": continue_cost,
        "intervene_cost": intervene_cost,
        "intervene_helped": intervene_helped,
        "branch_type": "offline_counterfactual",
    }


def generate_dataset(seeds: Iterable[int] = TRAINING_SEEDS) -> list[dict[str, Any]]:
    by_id = {task.task_id: task for task in TASKS}
    return [make_example(by_id[task_id], seed) for task_id in (*TRAIN_TASK_IDS, *EVALUATION_TASK_IDS) for seed in seeds]


def _fit_and_score(examples: list[dict[str, Any]]) -> tuple[Pipeline, dict[str, Any]]:
    train = [example for example in examples if example["task_id"] in TRAIN_TASK_IDS]
    evaluation = [example for example in examples if example["task_id"] in EVALUATION_TASK_IDS]
    y_train = np.asarray([example["intervene_helped"] for example in train], dtype=int)
    y_eval = np.asarray([example["intervene_helped"] for example in evaluation], dtype=int)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("training data must contain both intervene_helped classes")
    if len(set(y_eval.tolist())) < 2:
        raise ValueError("held-out evaluation tasks must contain both intervene_helped classes")
    x_train = np.asarray([[example["features"][name] for name in FEATURE_NAMES] for example in train], dtype=float)
    x_eval = np.asarray([[example["features"][name] for name in FEATURE_NAMES] for example in evaluation], dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_eval).all():
        raise ValueError("training/evaluation features must be finite")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(random_state=0, solver="liblinear", max_iter=1000)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    score = {
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(evaluation)),
        "train_class_counts": {str(label): int((y_train == label).sum()) for label in (0, 1)},
        "evaluation_class_counts": {str(label): int((y_eval == label).sum()) for label in (0, 1)},
        "evaluation_accuracy": float(accuracy_score(y_eval, predictions)),
        "evaluation_confusion_matrix_labels_0_1": confusion_matrix(y_eval, predictions, labels=[0, 1]).tolist(),
    }
    return model, score


def train_and_save(
    artifact_path: str | Path = ARTIFACT_PATH,
    dataset_path: str | Path = DATASET_PATH,
    seeds: Iterable[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    seeds = tuple(seeds)
    examples = generate_dataset(seeds)
    model, score = _fit_and_score(examples)
    task_order = [task.task_id for task in TASKS]
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "target": "intervene_helped",
        "feature_names": list(FEATURE_NAMES),
        "cost_matrix": COST_MATRIX,
        "decision_rule": "choose the action with lower expected cost from predict_proba; ties continue",
        "model": model,
        "training_metadata": {
            "algorithm": "StandardScaler + LogisticRegression",
            "risk_signal": RISK_SIGNAL_ID,
            "feature_generation": "run_demo.features_from_observation",
            "random_state": 0,
            "training_tasks": list(TRAIN_TASK_IDS),
            "evaluation_tasks": list(EVALUATION_TASK_IDS),
            "task_order": task_order,
            "seeds": list(seeds),
            "split_unit": "task_id",
            **score,
            "limitations": [
                "Synthetic arithmetic fixtures only; task-held-out scores are exploratory, not evidence of generalization.",
                "The offline risk feature is a neutral prior, not a critic or calibrated risk estimate.",
                "The label describes whether intervention passed while continuation failed; it does not capture every disruption cost.",
                "failure_prob is a neutral 0.5 prior because the current fixtures contain no validated, non-leaking pre-action risk signal.",
            ],
        },
    }
    artifact_path = Path(artifact_path)
    dataset_path = Path(dataset_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    with dataset_path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, sort_keys=True) + "\n")
    return {"artifact": artifact, "examples": examples, "artifact_path": artifact_path, "dataset_path": dataset_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    result = train_and_save(args.artifact, args.dataset)
    metadata = result["artifact"]["training_metadata"]
    print(f"Training examples: {len(result['examples'])}")
    print(f"Train tasks: {metadata['training_tasks']}; evaluation tasks: {metadata['evaluation_tasks']}")
    print(f"Train class counts: {metadata['train_class_counts']}; evaluation class counts: {metadata['evaluation_class_counts']}")
    print(f"Held-out evaluation accuracy: {metadata['evaluation_accuracy']:.3f} ({metadata['evaluation_rows']} examples; exploratory)")
    print(f"Saved artifact: {result['artifact_path']}")
    print(f"Saved offline branch records: {result['dataset_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
