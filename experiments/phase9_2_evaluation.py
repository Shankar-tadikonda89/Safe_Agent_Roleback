"""Phase 9.2 frozen offline training/evaluation pipeline.

Run ``--mode prepare`` first. It writes the preregistration and exact split
manifest before any held-out potential outcomes are generated. Evaluation
requires a separately verified, checksummed model artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_demo import FEATURE_NAMES
from experiments import phase9_2_simulator as sim

DEFAULT_PREREG = Path("results/phase9_2_preregistration.md")
DEFAULT_MANIFEST = Path("results/phase9_2_split_manifest.json")
DEFAULT_OUTPUT = Path("results/phase9_2_diverse_final")
DEFAULT_MODEL_DIR = Path("models/phase9_2_diverse_final")
REGULARIZATION_GRID = (0.1, 1.0, 10.0)
VARIANTS = {
    "full": tuple(FEATURE_NAMES),
    "failure_prob_only": ("failure_prob",),
    "drop_failure_prob": tuple(x for x in FEATURE_NAMES if x != "failure_prob"),
    "drop_reversibility_proxy": tuple(x for x in FEATURE_NAMES if x != "reversibility_proxy"),
    "drop_trajectory_position": tuple(x for x in FEATURE_NAMES if x != "trajectory_position"),
    "drop_disruption_history": tuple(x for x in FEATURE_NAMES if x != "disruption_history"),
}

PREREGISTRATION = """# Phase 9.2 preregistration — diverse synthetic mechanisms

Status: frozen design, before held-out potential outcomes are generated.
This is an offline synthetic mechanism study, not an LLM-agent experiment.

## Hypotheses and estimand

Primary estimand: paired difference in mean realized task cost for the full
four-feature policy versus each frozen baseline on the held-out family
configurations. A lower mean cost is favorable under the unchanged Phase 9
cost table. The policy may tie or lose; no result is guaranteed. Secondary
outcomes are task success/failure, selected recoveries, unnecessary actions,
ineffective actions, harmful transitions, intervention rate, executed steps,
and local policy runtime. No statistical-significance claim is planned.

## Outcome classes and cost objective (unchanged)

Class order: recovery, unnecessary_intervention, ineffective_intervention,
intervention_harm. Continue costs are 3/0/3/0 and intervene costs are 2/2/5/5.
Failure penalty is 3; intervention overhead cost is 2 and is included once in
the intervention costs. Expected continue cost is 3*(q_recovery +
q_ineffective); expected intervene cost is 2 + 3*(q_ineffective + q_harm).
Intervene only when strictly cheaper; ties continue. Abort is failure at cost
3 with no executed steps or intervention overhead.

## Mechanisms and observation model

Eight mechanisms are represented: transient repairable tool fault, stale
evidence, irreversible action, late recovery window, accumulated replanning
disruption, conflicting evidence, tool drift, and resource budget. These
mechanisms use different transition rules, not just different IDs. Each
family configuration has frozen values within the ranges in the split
manifest. Those ranges are synthetic design assumptions, not empirical
estimates. Held-out configurations include tool drift, resource budget, and
other parameter combinations absent from training; this tests simulator
transfer only.

Latent fragility, stale/conflict state, actual fault, repair draw, harm draw,
and branch outcomes are simulation truth. A noisy pre-action risk signal and
an alert with family-specific sensitivity/false-positive probability form
failure_prob. Alert false negatives/positives are possible. The alert is
generated from fragility/evidence state, never from the realized fault bit.
Reversibility comes from visible action metadata; position is current visible
step over planned steps; disruption history counts only visible prior replans.
The inference allowlist and order are failure_prob, reversibility_proxy,
trajectory_position, disruption_history. IDs, seeds, oracle labels, latent
state, faults, harm draws, and potential outcomes are excluded.

The harm rate is illustrative. Base settings are 0%, 5%, 10%, and 20% for
sensitivity; family/action/history multipliers are documented simulator
mechanics. Potential outcomes are paired synthetic counterfactuals, not
simultaneously observed outcomes.

## Frozen splits and model procedure

The split manifest records all family configurations and scenario keys before
held-out outcome generation. Targets are 12 training, 6 validation, 4 pilot,
and 30 held-out family configurations, each with 256 scenarios. Family IDs
are disjoint. Root seeds are 20261201 (train), 20261202 (validation),
20261203 (pilot), and 20261204 (test); random streams are SHA-256-derived
and domain-separated. These counts are design targets, not a power analysis.

Fit StandardScaler plus multinomial LogisticRegression on training families
only. Select C from {0.1, 1, 10} by validation multiclass log loss, ties to
the smaller C. This fixed grid applies separately to each preregistered
feature ablation. No threshold or cost tuning is performed. Train the frozen
artifacts, verify and load them in a separate offline process, then evaluate.
Evaluation must not fit, tune, or retrain.

## Baselines and analysis

Compare no_intervention, naive_threshold (abort iff failure_prob > 0.5),
proposed, pilot_test, and prefix_branching_approximation on identical held-out
scenario keys and isolated initial states. The pilot baseline selects one
global action from a disjoint pilot set by lower mean frozen cost (ties
continue), then freezes it. The prefix-branching approximation executes
four simulator forecasts for each action from the same prefix, chooses the
lower estimated cost (ties continue), and logs eight branch forecasts and
their local runtime/overhead separately. It is an offline simulator-assisted
approximation, not a faithful reproduction or deployable policy. Branch
forecasts use streams separate from evaluation potential outcomes. Branch
compute is reported as resource overhead and is not silently added to the
unchanged Phase 9 realized-cost table.

Primary comparisons use paired cost differences. Confidence intervals are
95% percentile intervals from 10,000 resamples of held-out family-configuration
means, seed 90210. There are 30 family configurations; these are synthetic
mechanism configurations, not independent real-world domains. Report
per-mechanism outcomes and this limitation.

Metrics: success rate = verifier-passing selected outcomes / attempts;
failure rate = failures / attempts; recovery rate = selected recovery actions
/ scenarios whose continuation potential fails; unnecessary rate = selected
unnecessary actions / scenarios whose continuation potential passes; harm
rate = selected harmful transitions / selected interventions; intervention
rate = interventions / attempts; mean realized cost and mean executed steps
per attempt. Undefined rates are null. Probability quality is computed from
the four-class proposed probabilities and evaluator labels: multiclass Brier
is mean sum of squared class errors; log loss clips at 1e-15 and renormalizes;
reliability uses ten equal-width bins; confusion uses argmax; undefined
precision/recall/F1 are null. Runtime is local simulation/policy runtime, not
provider latency.

## Ablation and integrity

Evaluate full features, failure_prob only, and each leave-one-feature-out
variant. All variants and the regularization grid are frozen here; validation
selects C and all held-out ablation results are reported. No preferred feature
set is selected from held-out results. Artifact metadata includes schema,
class/feature order, costs, split hash, and model settings. Checksums are
verified before loading. Probability types are validated before numeric
conversion; strings, booleans, invalid shape/range/sum, and non-finite values
are rejected. A separate-process artifact check precedes held-out evaluation.

## Limitations

This study is fully synthetic; parameter ranges and harm multipliers are not
empirical. Mechanism families remain simulator abstractions. Results cannot
establish real-world task-domain generalization, LLM critic calibration,
cross-model transfer, or production effectiveness. The pilot and prefix
comparators have different information budgets, which are reported. No
held-out result may alter features, costs, family definitions, model choice,
or analysis. If the learned policy never intervenes or fails to improve,
report that result unchanged.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_design(prereg_path: Path = DEFAULT_PREREG, manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Freeze design files only; does not generate scenarios or outcomes."""
    if prereg_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a preregistration or split manifest")
    free_bytes = shutil.disk_usage(prereg_path.parent.resolve()).free
    required_bytes = 100 * 1024 * 1024
    if free_bytes < required_bytes:
        raise RuntimeError("resource preflight failed: less than 100 MiB free for Phase 9.2 artifacts")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prereg_path.write_text(PREREGISTRATION, encoding="utf-8")
    manifest = sim.split_manifest()
    manifest["preregistration_sha256"] = _sha256(prereg_path)
    manifest["resource_preflight"] = {
        "free_disk_bytes_at_freeze": free_bytes,
        "minimum_free_disk_bytes": required_bytes,
        "planned_training_validation_pilot_test_scenarios": sum(
            split["scenario_count"] for split in manifest["splits"].values()
        ),
        "heldout_policy_runs_primary": manifest["splits"]["test"]["scenario_count"] * len(sim.POLICY_NAMES),
        "note": "planning estimate only; not a power calculation or runtime benchmark",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"preregistration": prereg_path, "manifest": manifest_path,
            "manifest_sha256": _sha256(manifest_path), "heldout_scenario_count": manifest["splits"]["test"]["scenario_count"]}


def _load_frozen_design(prereg_path: Path, manifest_path: Path) -> tuple[str, dict[str, Any]]:
    if not prereg_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("frozen Phase 9.2 preregistration and manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != sim.EXPERIMENT_VERSION:
        raise ValueError("split manifest experiment version mismatch")
    if manifest.get("preregistration_sha256") != _sha256(prereg_path):
        raise ValueError("preregistration differs from the frozen manifest")
    # JSON round-trip normalizes immutable tuples (for example action_mix) to
    # the same lists stored in the frozen manifest.
    expected = json.loads(json.dumps(sim.split_manifest(), allow_nan=False))
    resource = manifest.pop("resource_preflight", None)
    expected["preregistration_sha256"] = manifest["preregistration_sha256"]
    if (manifest != expected or not isinstance(resource, dict)
            or not isinstance(resource.get("free_disk_bytes_at_freeze"), int)
            or resource.get("free_disk_bytes_at_freeze", 0) < resource.get("minimum_free_disk_bytes", 1)
            or resource.get("planned_training_validation_pilot_test_scenarios") != 12 * 256 + 6 * 256 + 4 * 256 + 30 * 256):
        raise ValueError("split manifest no longer matches the frozen family/seed design")
    return _sha256(manifest_path), manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{n}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{n} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _scenario_key(s: sim.Scenario) -> tuple[str, int, int]:
    return s.family.family_id, s.root_seed, s.seed


def _feature_matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    x = np.asarray([[r["features"][name] for name in names] for r in rows], dtype=float)
    if x.shape != (len(rows), len(names)) or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("feature matrix violates the schema or bounds")
    return x


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    y = np.asarray([r["outcome_class_index"] for r in rows], dtype=int)
    if not np.array_equal(np.unique(y), np.arange(len(sim.CLASS_NAMES))):
        missing = [sim.CLASS_NAMES[i] for i in range(4) if i not in set(y.tolist())]
        raise ValueError(f"training split missing outcome classes: {missing}")
    return y


def _new_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c_value, solver="lbfgs", random_state=0, max_iter=2500)),
    ])


def _strict_probability_matrix(raw: Any) -> np.ndarray:
    """Validate original estimator values before any float coercion."""
    try:
        values = np.asarray(raw, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability output has invalid structure") from exc
    if values.shape != (1, len(sim.CLASS_NAMES)):
        raise ValueError("probability output has invalid shape")
    for value in values[0]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("probability values must be numeric and not boolean")
    result = values.astype(float)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=sim.PROBABILITY_SUM_TOLERANCE):
        raise ValueError("probabilities must sum to one")
    return result


def _strict_probability_vector(raw: Any) -> np.ndarray:
    try:
        values = np.asarray(raw, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability vector has invalid structure") from exc
    if values.shape != (len(sim.CLASS_NAMES),):
        raise ValueError("probability vector has invalid shape")
    return _strict_probability_matrix(values.reshape(1, -1))[0]


def calculate_expected_costs(probabilities: Any) -> dict[str, Any]:
    q = _strict_probability_matrix(np.asarray(probabilities, dtype=object).reshape(1, -1))[0]
    q_r, _q_u, q_i, q_h = (float(v) for v in q)
    c = 3.0 * (q_r + q_i)
    i = 2.0 + 3.0 * (q_i + q_h)
    return {"class_names": list(sim.CLASS_NAMES), "class_probabilities": q.tolist(),
            "expected_continue_cost": c, "expected_intervene_cost": i,
            "selected_action": "intervene" if i < c else "continue"}


class ProposedPolicy:
    def __init__(self, model: Pipeline, feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)):
        if not np.array_equal(np.asarray(getattr(model, "classes_", [])), np.arange(len(sim.CLASS_NAMES))):
            raise ValueError("model class order is invalid")
        if tuple(feature_names) not in set(VARIANTS.values()):
            raise ValueError("model feature schema is not preregistered")
        self.model, self.feature_names = model, tuple(feature_names)

    def decide(self, features: dict[str, float]) -> dict[str, Any]:
        if tuple(features) != tuple(FEATURE_NAMES):
            raise ValueError("inference requires the exact ordered four-feature allowlist")
        x = np.asarray([[features[n] for n in self.feature_names]], dtype=float)
        if x.shape != (1, len(self.feature_names)) or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise ValueError("inference features are invalid")
        diagnostics = calculate_expected_costs(_strict_probability_matrix(self.model.predict_proba(x))[0])
        return diagnostics


def _training_examples(configs: list[sim.FamilyConfig], root_seed: int, harm_probability: float) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        for seed in range(sim.SCENARIOS_PER_FAMILY):
            s = sim.generate_scenario(config, seed, root_seed, harm_probability)
            outcome = sim.potential_outcomes(s)
            label = sim.CLASS_TO_INDEX[outcome["outcome_class"]]
            rows.append({
                "scenario_key": sim.scenario_key(s), "mechanism": config.mechanism,
                "feature_names": list(FEATURE_NAMES), "features": dict(s.features),
                "outcome_class_index": label, "outcome_class": sim.CLASS_NAMES[label],
                "potential_outcomes": outcome,
                "scenario_metadata": {"latent_fragility": s.latent_fragility,
                                      "fault_enabled": s.fault_enabled,
                                      "stale_evidence": s.stale_evidence,
                                      "conflicting_evidence": s.conflicting_evidence},
            })
    return rows


def _validation_loss(model: Pipeline, rows: list[dict[str, Any]], names: tuple[str, ...]) -> float:
    probabilities = _strict_probability_matrix_batch(model.predict_proba(_feature_matrix(rows, names)))
    return float(log_loss([r["outcome_class_index"] for r in rows], probabilities,
                          labels=list(range(len(sim.CLASS_NAMES)))))


def _strict_probability_matrix_batch(raw: Any) -> np.ndarray:
    try:
        values = np.asarray(raw, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability output has invalid structure") from exc
    if values.ndim != 2 or values.shape[1] != len(sim.CLASS_NAMES):
        raise ValueError("probability output has invalid shape")
    for row in values:
        _strict_probability_matrix(row.reshape(1, -1))
    return values.astype(float)


def _artifact_paths(model_dir: Path, variant: str) -> tuple[Path, Path]:
    artifact = model_dir / f"phase9_2_{variant}.joblib"
    return artifact, Path(str(artifact) + ".metadata.json")


def _save_artifact(model: Pipeline, metadata: dict[str, Any], model_dir: Path, variant: str) -> tuple[Path, str]:
    artifact, sidecar = _artifact_paths(model_dir, variant)
    if artifact.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite Phase 9.2 model {artifact}")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact)
    checksum = _sha256(artifact)
    sidecar.write_text(json.dumps({"artifact_sha256": checksum, "model_metadata": metadata},
                                  indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return artifact, checksum


def _load_artifact(artifact: Path) -> tuple[Pipeline, dict[str, Any], str]:
    _, sidecar = artifact, Path(str(artifact) + ".metadata.json")
    if not artifact.is_file() or not sidecar.is_file():
        raise FileNotFoundError("frozen Phase 9.2 model or metadata is missing")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    checksum = _sha256(artifact)
    if checksum != data.get("artifact_sha256"):
        raise ValueError("model checksum mismatch")
    metadata = data.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("experiment_version") != sim.EXPERIMENT_VERSION:
        raise ValueError("model metadata is invalid")
    if metadata.get("class_names") != list(sim.CLASS_NAMES):
        raise ValueError("metadata class order is invalid")
    expected_names = tuple(metadata.get("feature_names", ()))
    if expected_names not in set(VARIANTS.values()):
        raise ValueError("metadata feature order is invalid")
    payload = joblib.load(artifact)
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, Pipeline) or payload.get("metadata") != metadata:
        raise ValueError("model payload does not match metadata")
    if not np.array_equal(np.asarray(model.classes_), np.arange(len(sim.CLASS_NAMES))):
        raise ValueError("estimator class order is invalid")
    if len(model.named_steps) != 2 or not isinstance(model.named_steps.get("scale"), StandardScaler) or not isinstance(model.named_steps.get("classifier"), LogisticRegression):
        raise ValueError("fitted pipeline type differs from preregistration")
    if int(model.n_features_in_) != len(expected_names):
        raise ValueError("fitted pipeline feature count differs from metadata")
    return model, metadata, checksum


def train_and_save(prereg: Path, manifest_path: Path, output_dir: Path, model_dir: Path) -> dict[str, Any]:
    manifest_sha, manifest = _load_frozen_design(prereg, manifest_path)
    if output_dir.exists() or model_dir.exists():
        raise FileExistsError("Phase 9.2 training output/model path already exists; choose an unused path")
    splits = sim.family_configurations()
    training = _training_examples(splits["training"], sim.TRAIN_ROOT_SEED, sim.PRIMARY_HARM_PROBABILITY)
    validation = _training_examples(splits["validation"], sim.VALIDATION_ROOT_SEED, sim.PRIMARY_HARM_PROBABILITY)
    pilot = _training_examples(splits["pilot"], sim.PILOT_ROOT_SEED, sim.PRIMARY_HARM_PROBABILITY)
    y = _labels(training)
    validation_y = np.asarray([r["outcome_class_index"] for r in validation], dtype=int)
    if set(y.tolist()) != set(range(4)):
        raise ValueError("required training class missing; stopped before held-out generation")
    # Pilot baseline selection is isolated from model fitting and test families.
    pilot_continue = statistics.mean(r["potential_outcomes"]["continue_realized_cost"] for r in pilot)
    pilot_intervene = statistics.mean(r["potential_outcomes"]["intervene_realized_cost"] for r in pilot)
    pilot_action = "intervene" if pilot_intervene < pilot_continue else "continue"
    selected: dict[str, tuple[Pipeline, float, float]] = {}
    for variant, names in VARIANTS.items():
        x_train, x_val = _feature_matrix(training, names), _feature_matrix(validation, names)
        candidates = []
        for c_value in REGULARIZATION_GRID:
            model = _new_model(c_value)
            model.fit(x_train, y)
            if not np.array_equal(model.classes_, np.arange(4)):
                raise ValueError("training produced an invalid class order")
            score = _validation_loss(model, validation, names)
            candidates.append((score, c_value, model))
        score, c_value, model = min(candidates, key=lambda item: (item[0], item[1]))
        selected[variant] = (model, c_value, score)
    metadata_by_variant = {}
    checksums = {}
    for variant, names in VARIANTS.items():
        model, c_value, score = selected[variant]
        metadata = {
            "schema_version": sim.SCHEMA_VERSION,
            "experiment_version": sim.EXPERIMENT_VERSION,
            "feature_names": list(names), "class_names": list(sim.CLASS_NAMES),
            "costs": {"continue": sim.CONTINUE_COSTS, "intervene": sim.INTERVENE_COSTS,
                      "failure_penalty": sim.FAILURE_PENALTY,
                      "intervention_overhead_cost": sim.INTERVENTION_OVERHEAD_COST},
            "split_manifest_sha256": manifest_sha,
            "training_families": [r.family_id for r in splits["training"]],
            "validation_families": [r.family_id for r in splits["validation"]],
            "heldout_families": [r.family_id for r in splits["test"]],
            "training_class_counts": {name: int(np.sum(y == i)) for i, name in enumerate(sim.CLASS_NAMES)},
            "validation_log_loss": score, "selected_C": c_value,
            "selection_grid": list(REGULARIZATION_GRID),
            "model": {"pipeline": ["StandardScaler", "LogisticRegression"],
                      "solver": "lbfgs", "random_state": 0, "max_iter": 2500,
                      "class_weight": None, "sklearn_version": sklearn.__version__},
            "variant": variant,
        }
        metadata_by_variant[variant] = metadata
        _, checksums[variant] = _save_artifact(model, metadata, model_dir, variant)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "training_examples.jsonl", training)
    _write_jsonl(output_dir / "validation_examples.jsonl", validation)
    _write_jsonl(output_dir / "pilot_examples.jsonl", pilot)
    pilot_path = output_dir / "pilot_decision.json"
    pilot_path.write_text(json.dumps({
        "action": pilot_action, "mean_continue_cost": pilot_continue,
        "mean_intervene_cost": pilot_intervene, "tie_rule": "continue",
        "pilot_scenarios": len(pilot),
        "pilot_families": [r.family_id for r in splits["pilot"]],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "experiment_version": sim.EXPERIMENT_VERSION,
        "manifest_sha256": manifest_sha, "preregistration_sha256": manifest["preregistration_sha256"],
        "training_count": len(training), "validation_count": len(validation),
        "pilot_count": len(pilot), "training_class_counts": metadata_by_variant["full"]["training_class_counts"],
        "validation_class_counts": {name: int(np.sum(validation_y == i)) for i, name in enumerate(sim.CLASS_NAMES)},
        "pilot_action": pilot_action, "variants": {
            name: {"artifact": str(_artifact_paths(model_dir, name)[0]),
                   "sha256": checksums[name], "selected_C": selected[name][1],
                   "validation_log_loss": selected[name][2]} for name in VARIANTS
        },
        "heldout_outcomes_generated": False,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def verify_artifacts(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load and validate locally generated artifacts; intended as a separate process."""
    checks = {}
    for variant, names in VARIANTS.items():
        artifact, _ = _artifact_paths(model_dir, variant)
        model, metadata, checksum = _load_artifact(artifact)
        probe = {name: 0.37 for name in FEATURE_NAMES}
        x = np.asarray([[probe[n] for n in names]], dtype=float)
        probabilities = _strict_probability_matrix_batch(model.predict_proba(x))[0].tolist()
        checks[variant] = {
            "sha256": checksum, "feature_names": list(names),
            "class_names": [int(value) for value in model.named_steps["classifier"].classes_],
            "metadata_class_names": metadata["class_names"],
            "pipeline": [type(model.named_steps["scale"]).__name__,
                         type(model.named_steps["classifier"]).__name__],
            "probe_probabilities": probabilities,
        }
    report = {"experiment_version": sim.EXPERIMENT_VERSION,
              "variants": checks, "verification_process": "fresh offline CLI process"}
    report_path = output_dir / "artifact_verification.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite artifact verification report")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _family_branch_decision(scenario: sim.Scenario, harm_probability: float) -> tuple[str, dict[str, Any]]:
    estimates = {}
    for action in ("continue", "intervene"):
        passes = [sim.prefix_branch_rollout_passes(scenario, action, i, harm_probability)
                  for i in range(sim.PREFIX_BRANCH_ROLLOUTS_PER_ACTION)]
        probability = statistics.mean(passes)
        estimates[action] = ((3.0 * (1.0 - probability)) if action == "continue"
                             else 2.0 + 3.0 * (1.0 - probability))
    decision = "intervene" if estimates["intervene"] < estimates["continue"] else "continue"
    return decision, {"estimated_continue_cost": estimates["continue"],
                      "estimated_intervene_cost": estimates["intervene"],
                      "branch_rollouts": 2 * sim.PREFIX_BRANCH_ROLLOUTS_PER_ACTION,
                      "branch_overhead_steps": 2 * sim.PREFIX_BRANCH_ROLLOUTS_PER_ACTION,
                      "baseline_label": "simulator_assisted_prefix_branching_approximation"}


def _select_decision(s: sim.Scenario, name: str, model: ProposedPolicy | None,
                     pilot_action: str, harm_probability: float) -> tuple[str, dict[str, Any] | None]:
    if name == "no_intervention":
        return "continue", None
    if name == "naive_threshold":
        return ("abort" if s.features["failure_prob"] > 0.5 else "continue"), None
    if name == "proposed":
        if model is None:
            raise ValueError("proposed policy requires a frozen model")
        result = model.decide(dict(s.features))
        return result["selected_action"], result
    if name == "pilot_test":
        if pilot_action not in {"continue", "intervene"}:
            raise ValueError("invalid frozen pilot decision")
        return pilot_action, None
    if name == "prefix_branching_approximation":
        return _family_branch_decision(s, harm_probability)
    raise ValueError(f"unknown policy {name}")


def _public_run_prefix(s: sim.Scenario, policy: str, decision: str,
                       diagnostics: dict[str, Any] | None, elapsed: float) -> dict[str, Any]:
    row = {
        "schema_version": sim.SCHEMA_VERSION, "experiment_version": sim.EXPERIMENT_VERSION,
        "scenario_key": sim.scenario_key(s), "policy": policy,
        "feature_names": list(FEATURE_NAMES), "features": dict(s.features),
        "decision": decision, "decision_diagnostics": diagnostics,
        "initial_state": json.loads(json.dumps(s.initial_state)),
        "initial_state_hash": hashlib.sha256(json.dumps(s.initial_state, sort_keys=True).encode()).hexdigest(),
        "local_latency_seconds": elapsed, "mode": "synthetic_offline", "llm_calls": 0,
    }
    if policy == "prefix_branching_approximation" and diagnostics:
        row["branch_diagnostics"] = diagnostics
    return row


def _run_scenarios(scenarios: list[sim.Scenario], model: ProposedPolicy,
                   pilot_action: str, harm_probability: float, model_sha: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Every policy decision, including independent prefix forecasts, is fixed before
    # evaluator-only potential_outcomes is called.
    decisions = []
    for scenario in scenarios:
        for policy in sim.POLICY_NAMES:
            start = time.perf_counter()
            decision, diagnostics = _select_decision(
                scenario, policy, model if policy == "proposed" else None,
                pilot_action, harm_probability,
            )
            decisions.append(_public_run_prefix(scenario, policy, decision, diagnostics,
                                                time.perf_counter() - start))
    potentials = []
    for scenario in scenarios:
        outcome = sim.potential_outcomes(scenario)
        potentials.append({"scenario_key": sim.scenario_key(scenario),
                           "potential_outcomes": outcome,
                           "scenario_metadata": {
                               "fault_enabled": scenario.fault_enabled,
                               "latent_fragility": scenario.latent_fragility,
                               "stale_evidence": scenario.stale_evidence,
                               "conflicting_evidence": scenario.conflicting_evidence,
                           }})
    potential_by_key = {_key_from_public(r["scenario_key"]): r["potential_outcomes"] for r in potentials}
    for row in decisions:
        po = potential_by_key[_key_from_public(row["scenario_key"])]
        action = row["decision"]
        if action == "continue":
            passed, cost, steps = po["continue_passed"], po["continue_realized_cost"], 1
        elif action == "intervene":
            passed, cost, steps = po["intervene_passed"], po["intervene_realized_cost"], 2
        elif action == "abort":
            passed, cost, steps = False, sim.FAILURE_PENALTY, 0
        else:
            raise ValueError("unsupported policy action")
        row.update({"selected_outcome_passed": bool(passed), "success": bool(passed),
                    "failure": not bool(passed), "realized_cost": float(cost),
                    "steps_executed": steps, "intervention": action == "intervene",
                    "intervention_overhead_steps": 1 if action == "intervene" else 0,
                    "model_sha256": model_sha if row["policy"] == "proposed" else None})
        if row["policy"] == "prefix_branching_approximation":
            row["branch_overhead_steps"] = row["branch_diagnostics"]["branch_overhead_steps"]
    return decisions, potentials


def _key_from_public(value: dict[str, Any]) -> tuple[str, int, int]:
    return value["family_id"], value["root_seed"], value["seed"]


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def _cluster_ci(values_by_family: dict[str, list[float]], seed: int = sim.BOOTSTRAP_SEED,
                resamples: int = sim.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    family_means = [statistics.mean(v) for _, v in sorted(values_by_family.items())]
    if len(family_means) < 2:
        return {"lower_95": None, "upper_95": None, "cluster_count": len(family_means),
                "resamples": resamples, "seed": seed}
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choices(family_means, k=len(family_means))) for _ in range(resamples)]
    return {"lower_95": _percentile(draws, 0.025), "upper_95": _percentile(draws, 0.975),
            "cluster_count": len(family_means), "resamples": resamples, "seed": seed,
            "resampling_unit": "heldout_family_configuration"}


def _outcome_metrics(rows: list[dict[str, Any]], po_by_key: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    interventions = [r for r in rows if r["intervention"]]
    selected = [po_by_key[_key_from_public(r["scenario_key"])]["outcome_class"] for r in interventions]
    class_counts = {name: selected.count(name) for name in sim.CLASS_NAMES}
    continue_passes = sum(bool(x["continue_passed"]) for x in po_by_key.values())
    continue_failures = n - continue_passes
    successes = sum(bool(r["success"]) for r in rows)
    return {
        "attempts": n, "successes": successes, "failures": n - successes,
        "success_rate": successes / n if n else None,
        "failure_rate": (n - successes) / n if n else None,
        "mean_realized_cost": statistics.mean(r["realized_cost"] for r in rows) if n else None,
        "intervention_count": len(interventions), "intervention_rate": len(interventions) / n if n else None,
        "abort_count": sum(r["decision"] == "abort" for r in rows),
        "recovery_count": class_counts["recovery"], "recovery_denominator_continue_failures": continue_failures,
        "recovery_rate": class_counts["recovery"] / continue_failures if continue_failures else None,
        "unnecessary_count": class_counts["unnecessary_intervention"],
        "unnecessary_denominator_continue_passes": continue_passes,
        "unnecessary_rate": class_counts["unnecessary_intervention"] / continue_passes if continue_passes else None,
        "ineffective_count": class_counts["ineffective_intervention"],
        "ineffective_rate_per_intervention": class_counts["ineffective_intervention"] / len(interventions) if interventions else None,
        "harm_count": class_counts["intervention_harm"],
        "harm_rate_per_intervention": class_counts["intervention_harm"] / len(interventions) if interventions else None,
        "overhead_steps": sum(r["intervention_overhead_steps"] for r in rows),
        "branch_overhead_steps": sum(r.get("branch_overhead_steps", 0) for r in rows),
        "branch_rollouts": sum(r.get("branch_diagnostics", {}).get("branch_rollouts", 0) for r in rows),
        "average_steps": statistics.mean(r["steps_executed"] for r in rows) if n else None,
        "average_local_policy_runtime_seconds": statistics.mean(r["local_latency_seconds"] for r in rows) if n else None,
        "selected_intervention_outcome_counts": class_counts,
    }


def _probability_metrics(rows: list[dict[str, Any]], po_by_key: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    proposed = {_key_from_public(r["scenario_key"]): r for r in rows if r["policy"] == "proposed"}
    keys = sorted(proposed)
    q = [_strict_probability_vector(proposed[k]["decision_diagnostics"]["class_probabilities"]) for k in keys]
    actual = [sim.CLASS_TO_INDEX[po_by_key[k]["outcome_class"]] for k in keys]
    n = len(keys)
    brier = statistics.mean(sum((float(qi[j]) - float(actual[i] == j)) ** 2 for j in range(4))
                            for i, qi in enumerate(q))
    clipped = np.maximum(np.asarray(q, dtype=float), sim.LOG_LOSS_EPSILON)
    clipped /= clipped.sum(axis=1, keepdims=True)
    loss = statistics.mean(-math.log(float(clipped[i, actual[i]])) for i in range(n))
    predicted = [int(np.argmax(qi)) for qi in q]
    matrix = [[sum(actual[i] == a and predicted[i] == p for i in range(n)) for p in range(4)] for a in range(4)]
    per_class = {}
    for j, name in enumerate(sim.CLASS_NAMES):
        tp = matrix[j][j]
        pp, ap = sum(matrix[a][j] for a in range(4)), sum(matrix[j])
        f1den = 2 * tp + (pp - tp) + (ap - tp)
        bins = []
        probs = [float(v[j]) for v in q]
        for b in range(10):
            members = [(probs[i], int(actual[i] == j)) for i in range(n)
                       if b / 10 <= probs[i] < (b + 1) / 10 or (b == 9 and probs[i] == 1)]
            bins.append({"low": b / 10, "high": (b + 1) / 10, "count": len(members),
                         "mean_predicted": statistics.mean(x for x, _ in members) if members else None,
                         "observed_rate": statistics.mean(y for _, y in members) if members else None})
        per_class[name] = {
            "precision": tp / pp if pp else None, "recall": tp / ap if ap else None,
            "f1": 2 * tp / f1den if f1den else None,
            "mean_predicted_probability": statistics.mean(probs),
            "observed_frequency": sum(y == j for y in actual) / n,
            "argmax_frequency": sum(y == j for y in predicted) / n,
            "reliability_bins": bins,
        }
    return {"multiclass_brier_score": brier, "multiclass_log_loss": loss,
            "class_order": list(sim.CLASS_NAMES),
            "mean_predicted_probabilities": {k: v["mean_predicted_probability"] for k, v in per_class.items()},
            "observed_class_frequencies": {k: v["observed_frequency"] for k, v in per_class.items()},
            "confusion_matrix_actual_rows_argmax_columns": matrix, "per_class": per_class,
            "log_loss_clip_epsilon": sim.LOG_LOSS_EPSILON}


def calculate_metrics(rows: list[dict[str, Any]], potential_rows: list[dict[str, Any]]) -> dict[str, Any]:
    po = {_key_from_public(r["scenario_key"]): r["potential_outcomes"] for r in potential_rows}
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_key_from_public(row["scenario_key"]), {})[row["policy"]] = row
    if set(grouped) != set(po) or any(set(x) != set(sim.POLICY_NAMES) for x in grouped.values()):
        raise ValueError("metrics require exactly one row per policy per matched scenario")
    summaries, by_mechanism = {}, {}
    for policy in sim.POLICY_NAMES:
        policy_rows = [grouped[k][policy] for k in sorted(grouped)]
        summaries[policy] = _outcome_metrics(policy_rows, po)
        by_mechanism[policy] = {}
        mechanisms = sorted({r["scenario_key"]["mechanism"] for r in policy_rows})
        for mechanism in mechanisms:
            selected = [r for r in policy_rows if r["scenario_key"]["mechanism"] == mechanism]
            selected_po = {_key_from_public(r["scenario_key"]): po[_key_from_public(r["scenario_key"])] for r in selected}
            by_mechanism[policy][mechanism] = _outcome_metrics(selected, selected_po)
    comparisons = {}
    for baseline in sim.POLICY_NAMES:
        if baseline == "proposed":
            continue
        by_family: dict[str, list[float]] = {}
        deltas = []
        for key, policies in grouped.items():
            delta = policies["proposed"]["realized_cost"] - policies[baseline]["realized_cost"]
            deltas.append(delta)
            by_family.setdefault(key[0], []).append(delta)
        comparisons[f"proposed_minus_{baseline}"] = {
            "mean_cost_difference": statistics.mean(deltas),
            "confidence_interval_95": _cluster_ci(by_family),
        }
    return {
        "experiment_version": sim.EXPERIMENT_VERSION,
        "scenario_count": len(grouped), "policy_run_count": len(rows),
        "class_order": list(sim.CLASS_NAMES),
        "costs": {"continue": sim.CONTINUE_COSTS, "intervene": sim.INTERVENE_COSTS},
        "policy_metrics": summaries, "paired_cost_differences": comparisons,
        "probability_quality": _probability_metrics(rows, po),
        "by_mechanism": by_mechanism,
        "limitations": ["synthetic simulator only", "family configurations are not real-world domains",
                        "local timing is variable and not provider latency",
                        "prefix branching is a simulator-assisted approximation"],
    }


def _validate_rows(rows: list[dict[str, Any]], potentials: list[dict[str, Any]]) -> None:
    po = {_key_from_public(r["scenario_key"]): r["potential_outcomes"] for r in potentials}
    if len(po) != len(potentials):
        raise ValueError("duplicate potential-outcome scenario key")
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("feature_names") != list(FEATURE_NAMES) or tuple(row.get("features", {})) != tuple(FEATURE_NAMES):
            raise ValueError("policy row has invalid feature schema")
        grouped.setdefault(_key_from_public(row["scenario_key"]), {})[row["policy"]] = row
    if set(grouped) != set(po) or any(set(v) != set(sim.POLICY_NAMES) for v in grouped.values()):
        raise ValueError("policy rows are not exactly matched")
    for key, policies in grouped.items():
        reference = next(iter(policies.values()))
        for row in policies.values():
            if (row["features"] != reference["features"] or row["initial_state_hash"] != reference["initial_state_hash"]
                    or row["scenario_key"] != reference["scenario_key"]):
                raise ValueError("policy scenario or initial state mismatch")
            outcome = po[key]
            if row["decision"] == "continue":
                expected = (outcome["continue_passed"], outcome["continue_realized_cost"], 1)
            elif row["decision"] == "intervene":
                expected = (outcome["intervene_passed"], outcome["intervene_realized_cost"], 2)
            else:
                expected = (False, sim.FAILURE_PENALTY, 0)
            if (row["success"], row["realized_cost"], row["steps_executed"]) != expected:
                raise ValueError("selected outcome/cost does not match the potential outcome")
        proposed = policies["proposed"]
        diag = proposed["decision_diagnostics"]
        expected_action = calculate_expected_costs(diag["class_probabilities"])["selected_action"]
        if expected_action != proposed["decision"]:
            raise ValueError("proposed decision diagnostics disagree with selected action")


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _create_test_scenarios(harm_probability: float) -> list[sim.Scenario]:
    configs = sim.family_configurations()["test"]
    return [sim.generate_scenario(config, seed, sim.TEST_ROOT_SEED, harm_probability)
            for config in configs for seed in range(sim.SCENARIOS_PER_FAMILY)]


def _load_verified_models(model_dir: Path, output_dir: Path, manifest_sha: str) -> tuple[dict[str, ProposedPolicy], dict[str, str], str]:
    report_path = output_dir / "artifact_verification.json"
    training_path = output_dir / "training_manifest.json"
    if not report_path.is_file() or not training_path.is_file():
        raise FileNotFoundError("separate-process artifact verification and training manifest are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    training_manifest = json.loads(training_path.read_text(encoding="utf-8"))
    if training_manifest.get("manifest_sha256") != manifest_sha:
        raise ValueError("training artifact was produced from a different split manifest")
    models, hashes = {}, {}
    for variant, names in VARIANTS.items():
        model, meta, checksum = _load_artifact(_artifact_paths(model_dir, variant)[0])
        check = report.get("variants", {}).get(variant, {})
        if check.get("sha256") != checksum or meta.get("feature_names") != list(names):
            raise ValueError("separately verified artifact differs from evaluation model")
        if meta.get("split_manifest_sha256") != manifest_sha:
            raise ValueError("model split provenance mismatch")
        models[variant], hashes[variant] = ProposedPolicy(model, names), checksum
    return models, hashes, training_manifest["pilot_action"]


def _write_policy_outputs(output_dir: Path, rows: list[dict[str, Any]], potentials: list[dict[str, Any]],
                          metrics: dict[str, Any], manifest_sha: str, model_sha: str,
                          harm_probability: float, sensitivity: bool = False) -> None:
    _validate_rows(rows, potentials)
    stem = "sensitivity_" + f"{harm_probability:.2f}" if sensitivity else "evaluation"
    _write_jsonl(output_dir / f"{stem}_runs.jsonl", rows)
    _write_jsonl(output_dir / f"{stem}_potential_outcomes.jsonl", potentials)
    _write_metrics(output_dir / f"{stem}_metrics.json", metrics)
    manifest_path = output_dir / f"{stem}_run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps({
        "experiment_version": sim.EXPERIMENT_VERSION, "mode": "synthetic_offline",
        "harm_probability": harm_probability, "split_manifest_sha256": manifest_sha,
        "model_sha256": model_sha, "scenario_count": len(potentials),
        "policy_run_count": len(rows), "policy_names": list(sim.POLICY_NAMES),
        "heldout_outcomes_generated_after_decisions": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ablation_metrics(output_dir: Path, scenarios: list[sim.Scenario], potentials: list[dict[str, Any]],
                      models: dict[str, ProposedPolicy], hashes: dict[str, str]) -> dict[str, Any]:
    po = {_key_from_public(r["scenario_key"]): r["potential_outcomes"] for r in potentials}
    all_rows = []
    result = {}
    for variant in VARIANTS:
        rows = []
        for s in scenarios:
            start = time.perf_counter()
            diag = models[variant].decide(dict(s.features))
            elapsed = time.perf_counter() - start
            decision = diag["selected_action"]
            outcome = po[_scenario_key(s)]
            passed = outcome["continue_passed"] if decision == "continue" else outcome["intervene_passed"]
            cost = outcome["continue_realized_cost"] if decision == "continue" else outcome["intervene_realized_cost"]
            rows.append({"scenario_key": sim.scenario_key(s), "policy": variant,
                         "features": dict(s.features), "feature_names": list(FEATURE_NAMES),
                         "decision": decision, "decision_diagnostics": diag,
                         "success": bool(passed), "realized_cost": float(cost),
                         "steps_executed": 1 if decision == "continue" else 2,
                         "intervention": decision == "intervene",
                         "intervention_overhead_steps": 1 if decision == "intervene" else 0,
                         "branch_overhead_steps": 0, "local_latency_seconds": elapsed})
        all_rows.extend(rows)
        # Reuse the common outcome/probability calculations under the variant's own name.
        result[variant] = _ablation_summary(rows, po)
    _write_jsonl(output_dir / "ablation_runs.jsonl", all_rows)
    return result


def _ablation_summary(rows: list[dict[str, Any]], po: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    summary = _outcome_metrics(rows, po)
    n = len(rows)
    ys = [sim.CLASS_TO_INDEX[po[_key_from_public(r["scenario_key"])]["outcome_class"]] for r in rows]
    qs = [_strict_probability_vector(r["decision_diagnostics"]["class_probabilities"]) for r in rows]
    brier = statistics.mean(sum((float(q[j]) - float(y == j)) ** 2 for j in range(4)) for q, y in zip(qs, ys))
    loss = statistics.mean(-math.log(max(sim.LOG_LOSS_EPSILON, float(q[y]))) for q, y in zip(qs, ys))
    per_mechanism = {}
    for mechanism in sorted({r["scenario_key"]["mechanism"] for r in rows}):
        mr = [r for r in rows if r["scenario_key"]["mechanism"] == mechanism]
        mp = {_key_from_public(r["scenario_key"]): po[_key_from_public(r["scenario_key"])] for r in mr}
        per_mechanism[mechanism] = _outcome_metrics(mr, mp)
    return {"metrics": summary, "multiclass_brier_score": brier,
            "multiclass_log_loss": loss, "by_mechanism": per_mechanism}


def run_evaluation(prereg: Path, manifest_path: Path, output_dir: Path, model_dir: Path,
                   harm_probability: float = sim.PRIMARY_HARM_PROBABILITY,
                   sensitivity: bool = False,
                   verification_dir: Path | None = None) -> dict[str, Any]:
    manifest_sha, _ = _load_frozen_design(prereg, manifest_path)
    models, hashes, pilot_action = _load_verified_models(
        model_dir, verification_dir or output_dir, manifest_sha
    )
    model = models["full"]
    scenarios = _create_test_scenarios(harm_probability)
    rows, potentials = _run_scenarios(scenarios, model, pilot_action, harm_probability, hashes["full"])
    metrics = calculate_metrics(rows, potentials)
    _write_policy_outputs(output_dir, rows, potentials, metrics, manifest_sha, hashes["full"], harm_probability, sensitivity)
    if not sensitivity:
        ablations = _ablation_metrics(output_dir, scenarios, potentials, models, hashes)
        _write_metrics(output_dir / "ablation_metrics.json", {
            "experiment_version": sim.EXPERIMENT_VERSION,
            "heldout_scenarios": len(scenarios),
            "variants": ablations,
            "selection": "all variants frozen in preregistration; C selected on validation only",
        })
        summary = _render_summary(metrics, ablations, harm_probability)
        summary_path = output_dir / "comparison_summary.md"
        if summary_path.exists():
            raise FileExistsError("refusing to overwrite comparison summary")
        summary_path.write_text(summary, encoding="utf-8")
        design_path = output_dir / "scenario_design.md"
        if design_path.exists():
            raise FileExistsError("refusing to overwrite scenario design")
        design_path.write_text(_scenario_design_text(), encoding="utf-8")
    return {"scenario_count": len(scenarios), "policy_run_count": len(rows), "metrics": metrics}


def _scenario_design_text() -> str:
    return """# Phase 9.2 scenario design\n\n""" + PREREGISTRATION.split("## Mechanisms and observation model\n", 1)[1].split("## Frozen splits", 1)[0]


def _render_summary(metrics: dict[str, Any], ablations: dict[str, Any], harm_probability: float) -> str:
    lines = [
        "# Phase 9.2 diverse synthetic mechanisms",
        "",
        "Synthetic offline simulator results; no real LLM-agent performance is measured.",
        f"Harm base setting: {harm_probability:.0%} (illustrative). Scenarios: {metrics['scenario_count']}.",
        "",
        "| Policy | Successes / attempts | Mean cost | Interventions | Recovery | Unnecessary | Ineffective | Harm | Avg steps | Branch forecasts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, m in metrics["policy_metrics"].items():
        lines.append(f"| {policy} | {m['successes']}/{m['attempts']} | {m['mean_realized_cost']:.5f} | {m['intervention_count']} | {m['recovery_count']} | {m['unnecessary_count']} | {m['ineffective_count']} | {m['harm_count']} | {m['average_steps']:.4f} | {m['branch_rollouts']} |")
    lines.extend(["", "## Paired cost differences", ""])
    for baseline, data in metrics["paired_cost_differences"].items():
        ci = data["confidence_interval_95"]
        lines.append(f"- `{baseline}`: {data['mean_cost_difference']:.6f}; 95% family-configuration bootstrap [{ci['lower_95']:.6f}, {ci['upper_95']:.6f}] ({ci['cluster_count']} clusters, {ci['resamples']} resamples).")
    q = metrics["probability_quality"]
    lines.extend(["", "## Probability quality", "",
                  f"- Multiclass Brier: {q['multiclass_brier_score']:.6f}; log loss: {q['multiclass_log_loss']:.6f}.",
                  f"- Mean predicted probabilities: `{json.dumps(q['mean_predicted_probabilities'], sort_keys=True)}`.",
                  f"- Observed class frequencies: `{json.dumps(q['observed_class_frequencies'], sort_keys=True)}`.",
                  "- See `metrics.json` for reliability bins, confusion matrix, per-class metrics, and mechanism groups.",
                  "", "## Feature ablations", ""])
    for name, result in ablations.items():
        m = result["metrics"]
        lines.append(f"- `{name}`: mean cost {m['mean_realized_cost']:.5f}; success {m['successes']}/{m['attempts']}; interventions {m['intervention_count']}; Brier {result['multiclass_brier_score']:.6f}; log loss {result['multiclass_log_loss']:.6f}.")
    lines.extend(["", "## Interpretation limits", "",
                  "The family mechanisms are synthetic assumptions, not empirical tasks. The pilot baseline uses a separate pilot set. Prefix branching is a simulator-assisted approximation with eight logged rollout forecasts per scenario; its information and compute budget differ from the learned policy. Family-level intervals describe only these sampled configurations. No statistical-significance, real-world generalization, or LLM-agent effectiveness claim is supported.", ""])
    return "\n".join(lines)


def run_sensitivity(prereg: Path, manifest: Path, output_dir: Path, model_dir: Path,
                    harm_probability: float, verification_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if harm_probability not in sim.SENSITIVITY_HARM_PROBABILITIES:
        raise ValueError("harm rate is not preregistered")
    if output_dir.exists():
        raise FileExistsError("sensitivity output path exists; choose a new path")
    output_dir.mkdir(parents=True)
    return run_evaluation(prereg, manifest, output_dir, model_dir, harm_probability,
                          sensitivity=True, verification_dir=verification_dir)


def audit_results(output_dir: Path, model_dir: Path, sensitivity: float | None = None) -> dict[str, Any]:
    stem = "sensitivity_" + f"{sensitivity:.2f}" if sensitivity is not None else "evaluation"
    rows = _read_jsonl(output_dir / f"{stem}_runs.jsonl")
    potentials = _read_jsonl(output_dir / f"{stem}_potential_outcomes.jsonl")
    saved = json.loads((output_dir / f"{stem}_metrics.json").read_text(encoding="utf-8"))
    recomputed = calculate_metrics(rows, potentials)
    if saved != recomputed:
        raise ValueError("saved metrics do not exactly reproduce from JSONL")
    expected_n = sim.TEST_FAMILY_COUNT * sim.SCENARIOS_PER_FAMILY
    if len(potentials) != expected_n or len(rows) != expected_n * len(sim.POLICY_NAMES):
        raise ValueError("record counts differ from frozen design")
    run_manifest = json.loads((output_dir / f"{stem}_run_manifest.json").read_text(encoding="utf-8"))
    artifact, _ = _artifact_paths(model_dir, "full")
    if run_manifest.get("model_sha256") != _sha256(artifact):
        raise ValueError("run model hash differs from frozen artifact")
    return {"status": "PASS", "scenario_count": len(potentials), "policy_run_count": len(rows),
            "metrics_recomputed": True, "artifact_loaded": False,
            "proposed_interventions": recomputed["policy_metrics"]["proposed"]["intervention_count"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train", "verify-artifacts", "evaluate", "sensitivity", "audit"), required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--harm-probability", type=float)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare_design(args.preregistration, args.manifest)
        print(f"Phase 9.2 design frozen; heldout_scenarios={result['heldout_scenario_count']}; manifest_sha256={result['manifest_sha256']}")
    elif args.mode == "train":
        result = train_and_save(args.preregistration, args.manifest, args.output_dir, args.model_dir)
        print(f"Phase 9.2 training complete; training={result['training_count']} validation={result['validation_count']} pilot={result['pilot_count']} heldout_outcomes_generated={result['heldout_outcomes_generated']}")
    elif args.mode == "verify-artifacts":
        report = verify_artifacts(args.model_dir, args.output_dir)
        print(f"Phase 9.2 artifact verification PASS; variants={len(report['variants'])}")
    elif args.mode == "evaluate":
        result = run_evaluation(args.preregistration, args.manifest, args.output_dir, args.model_dir)
        print(f"Phase 9.2 evaluation complete; scenarios={result['scenario_count']} policy_runs={result['policy_run_count']}")
    elif args.mode == "sensitivity":
        if args.harm_probability is None:
            parser.error("--mode sensitivity requires --harm-probability")
        result = run_sensitivity(args.preregistration, args.manifest, args.output_dir, args.model_dir, args.harm_probability)
        print(f"Phase 9.2 sensitivity complete; harm_probability={args.harm_probability:.2f}; scenarios={result['scenario_count']}")
    else:
        result = audit_results(args.output_dir, args.model_dir, args.harm_probability)
        print(f"Phase 9.2 audit {result['status']}; scenarios={result['scenario_count']} policy_runs={result['policy_run_count']} model_loaded={result['artifact_loaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

