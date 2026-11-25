"""Phase 9.8 authorized-partition data generation and model fitting.

This module never generates pilot or held-out scenarios. Policy observations
and evaluator potential outcomes are persisted to separate JSONL files. The
frozen Phase 9.7 simulator remains unmodified; Phase 9.8 configuration values
are applied only in the evaluator-side potential-outcome function below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments import phase9_7_simulator as sim
from experiments.phase9_5_observations import FEATURE_ORDER, extract_features
from experiments.phase9_7_runner import PHASE97_COST_MATRIX


PROTOCOL_PATH = Path("results/phase9_8_protocol.md")
CONFIG_PATH = Path("results/phase9_8_configuration_table.json")
MANIFEST_PATH = Path("results/phase9_8_split_manifest.json")
DATA_ROOT = Path("results/phase9_8_training")
MODEL_ROOT = Path("models/phase9_8_final")
EXPECTED_CONFIG_SHA256 = "efc4caf9cf570038a772bb76df57215cc412e3fce0e207566119db6dd83a3963"
EXPECTED_MANIFEST_SHA256 = "9686e7d0a53b5ec261afd9f02cca4d3c80bd036e4d6a55399aaf19848c1eda99"
PARTITIONS = ("fit", "calibration", "validation")
EXPECTED_GROUPS = {"fit": 40, "calibration": 20, "validation": 10}
EXPECTED_CLASS_NAMES = sim.OUTCOME_CLASSES
CLASS_INDEX = {name: i for i, name in enumerate(EXPECTED_CLASS_NAMES)}
C_GRID = (0.1, 1.0, 10.0)
ARTIFACT_VERSION = "phase9.8-multinomial-cost-sensitive-v1"
VARIANTS = {
    "full_features": FEATURE_ORDER,
    "failure_prob_only": ("failure_prob",),
    "leave_out_failure_prob": ("reversibility_proxy", "trajectory_position", "disruption_history"),
    "leave_out_reversibility_proxy": ("failure_prob", "trajectory_position", "disruption_history"),
    "leave_out_trajectory_position": ("failure_prob", "reversibility_proxy", "disruption_history"),
    "leave_out_disruption_history": ("failure_prob", "reversibility_proxy", "trajectory_position"),
}


class FrozenDesignError(ValueError):
    """Frozen protocol or manifest did not pass validation."""


class Phase98DataError(ValueError):
    """Generated records violate the frozen Phase 9.8 contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_self_hash(document: dict[str, Any], field: str, expected: str) -> str:
    if document.get(field) != expected:
        raise FrozenDesignError(f"{field} does not match the approved hash")
    payload = dict(document)
    payload.pop(field, None)
    actual = _sha256_bytes(_canonical_bytes(payload))
    if actual != expected:
        raise FrozenDesignError(f"{field} content hash mismatch")
    return actual


def load_and_validate_design() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Verify protocol, table, manifest, partitions, groups, and seed blocks."""
    for path in (PROTOCOL_PATH, CONFIG_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FrozenDesignError(f"required frozen design file is missing: {path}")
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    configs = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config_hash = _verify_self_hash(
        configs, "configuration_table_sha256", EXPECTED_CONFIG_SHA256,
    )
    _verify_self_hash(manifest, "split_manifest_sha256", EXPECTED_MANIFEST_SHA256)
    protocol_hash = _sha256_bytes(protocol.encode("utf-8"))
    for frozen_value in (
        EXPECTED_CONFIG_SHA256, EXPECTED_MANIFEST_SHA256,
        "validation multiclass log loss", "ties", "smaller `C`",
        "`C ∈ {0.1, 1, 10}`", "StandardScaler", "multinomial",
    ):
        if frozen_value not in protocol:
            raise FrozenDesignError(f"protocol does not contain frozen requirement: {frozen_value}")
    if manifest.get("configuration_table_sha256") != config_hash:
        raise FrozenDesignError("manifest references a different configuration table")
    if configs.get("schema_version") != "phase9.8-configuration-table-v1":
        raise FrozenDesignError("unsupported configuration-table schema")
    if manifest.get("schema_version") != "phase9.8-split-manifest-v1":
        raise FrozenDesignError("unsupported split-manifest schema")
    if tuple(configs["generator"].get("mechanism_order", ())) != sim.MECHANISMS:
        raise FrozenDesignError("mechanism order differs from Phase 9.7")
    if tuple(configs["generator"].get("mechanism_order", ())) != tuple(manifest.get("mechanism_order", ())):
        raise FrozenDesignError("manifest mechanism order differs from configuration table")
    expected_seed_partitions = {"fit", "calibration", "validation", "pilot", "heldout"}
    if set(manifest.get("root_seeds", {})) != expected_seed_partitions:
        raise FrozenDesignError("frozen root-seed set is invalid")

    rows = configs.get("configurations")
    groups = manifest.get("groups")
    if not isinstance(rows, list) or len(rows) != 120 or not isinstance(groups, list) or len(groups) != 120:
        raise FrozenDesignError("expected exactly 120 configuration groups")
    row_by_id = {row.get("configuration_id"): row for row in rows}
    group_by_id = {group.get("configuration_id"): group for group in groups}
    if len(row_by_id) != 120 or len(group_by_id) != 120 or set(row_by_id) != set(group_by_id):
        raise FrozenDesignError("configuration IDs are duplicated or differ from manifest groups")
    if manifest.get("partition_counts") != {
        "fit": 40, "calibration": 20, "validation": 10, "pilot": 10, "heldout": 40,
    }:
        raise FrozenDesignError("frozen partition totals differ from protocol")
    if manifest.get("scenario_counts") != {
        "fit": 10240, "calibration": 5120, "validation": 2560,
        "pilot": 2560, "heldout": 10240,
    }:
        raise FrozenDesignError("frozen scenario totals differ from protocol")
    if manifest.get("scenarios_per_configuration") != 256:
        raise FrozenDesignError("scenarios per configuration differ from protocol")

    allowed_ids: dict[str, set[str]] = {partition: set() for partition in manifest["partition_counts"]}
    ranges: list[tuple[int, int, str, str]] = []
    for group_index, group in enumerate(groups):
        cid = group["configuration_id"]
        row = row_by_id[cid]
        part = group.get("partition")
        if part not in allowed_ids:
            raise FrozenDesignError(f"group has unknown partition: {cid}")
        if row.get("mechanism") != group.get("mechanism"):
            raise FrozenDesignError(f"configuration and manifest mechanism mismatch: {cid}")
        if group.get("scenario_count") != 256:
            raise FrozenDesignError(f"scenario count mismatch: {cid}")
        if group.get("partition_root_seed") != manifest["root_seeds"].get(part):
            raise FrozenDesignError(f"root seed does not match partition: {cid}")
        start = group.get("scenario_seed_start_inclusive")
        end = group.get("scenario_seed_end_inclusive")
        if start != group_index * 256 or end != start + 255:
            raise FrozenDesignError(f"scenario seed block is invalid: {cid}")
        ranges.append((start, end, cid, part))
        allowed_ids[part].add(cid)
        params = configs["generator"]["mechanisms_and_parameter_ranges"].get(row["mechanism"])
        order = configs["generator"]["parameter_order_by_mechanism"].get(row["mechanism"])
        if not isinstance(params, dict) or not isinstance(order, list):
            raise FrozenDesignError(f"configuration parameter schema missing: {cid}")
        if set(row.get("parameters", {})) != set(order) or set(order) != set(params):
            raise FrozenDesignError(f"configuration parameter names mismatch: {cid}")
        if set(row.get("stratified_bin_indices", {})) != set(order):
            raise FrozenDesignError(f"configuration strata metadata mismatch: {cid}")
        for name in order:
            value = row["parameters"][name]
            bounds = params[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or len(bounds) != 2
                    or not bounds[0] <= value <= bounds[1]):
                raise FrozenDesignError(f"parameter outside frozen range: {cid}.{name}")
    ranges.sort()
    if ranges[0][0] != 0 or ranges[-1][1] != 30719 or any(
        left[1] + 1 != right[0] for left, right in zip(ranges, ranges[1:])
    ):
        raise FrozenDesignError("scenario seed blocks are overlapping or incomplete")
    for partition, expected_count in EXPECTED_GROUPS.items():
        if len(allowed_ids[partition]) != expected_count:
            raise FrozenDesignError(f"group count mismatch for authorized partition {partition}")
    for mechanism in sim.MECHANISMS:
        for partition, count in (("fit", 8), ("calibration", 4), ("validation", 2),
                                 ("pilot", 2), ("heldout", 8)):
            actual = sum(
                group["mechanism"] == mechanism and group["partition"] == partition
                for group in groups
            )
            if actual != count:
                raise FrozenDesignError(f"mechanism/partition count mismatch: {mechanism}/{partition}")
    if any(allowed_ids[a] & allowed_ids[b] for i, a in enumerate(allowed_ids) for b in list(allowed_ids)[i+1:]):
        raise FrozenDesignError("configuration group appears in multiple partitions")
    # The generator and data writer below accept only the three authorized partitions.
    if tuple(p for p in PARTITIONS if p in allowed_ids) != PARTITIONS:
        raise FrozenDesignError("authorized partitions are absent")
    return configs, manifest, protocol_hash


def configured_potential_outcomes(policy: sim.PolicyScenario, evaluator: sim.EvaluatorCase,
                                  configuration: dict[str, Any]) -> sim.PotentialOutcomes:
    """Generate a paired evaluator result with the frozen config, never policy input."""
    if not isinstance(policy, sim.PolicyScenario) or not isinstance(evaluator, sim.EvaluatorCase):
        raise Phase98DataError("public scenario and evaluator case must remain separate typed inputs")
    if policy.scenario_id != evaluator.scenario_id:
        raise Phase98DataError("public and evaluator records are not scenario matched")
    mechanism = configuration.get("mechanism")
    if mechanism != evaluator.mechanism or mechanism not in sim.MECHANISMS:
        raise Phase98DataError("configuration does not match evaluator mechanism")
    feature = extract_features(policy.snapshot.observation)
    p = configuration["parameters"]
    z = evaluator.susceptibility
    r = feature.reversibility_proxy
    x = feature.trajectory_position
    d = feature.disruption_history
    failure = 0.05 + 0.70 * z
    if mechanism == "transient_tool_failure":
        repair, harm = p["repair_probability"], p["harm_probability"]
    elif mechanism == "stale_contradictory_evidence":
        failure += p["failure_increment"]
        repair, harm = p["repair_probability"], p["harm_probability"]
    elif mechanism == "irreversible_action":
        severity = 1.0 - r
        failure += p["failure_sensitivity"] * severity
        repair = p["repair_probability"]
        harm = p["harm_base"] + p["harm_sensitivity"] * severity
    elif mechanism == "late_trajectory_failure":
        failure += p["failure_position_slope"] * x
        repair = p["repair_intercept"] - p["repair_position_decline"] * x
        harm = p["harm_intercept"] + p["harm_position_slope"] * x
    else:  # replanning_disruption
        failure += p["failure_disruption_slope"] * d
        repair = p["repair_intercept"] - p["repair_disruption_decline"] * d
        harm = p["harm_intercept"] + p["harm_disruption_slope"] * d
    failure, repair, harm = (min(1.0, max(0.0, float(v))) for v in (failure, repair, harm))
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (failure, repair, harm)):
        raise Phase98DataError("configured evaluator probabilities are invalid")
    continued = evaluator.continuation_uniform >= failure
    before_harm = continued or evaluator.repair_uniform < repair
    intervened = before_harm and evaluator.harm_uniform >= harm
    outcome_class = sim.classify_outcome(continued, intervened)
    return sim.PotentialOutcomes(
        continue_passed=continued,
        intervene_passed=intervened,
        outcome_class=outcome_class,
        continue_realized_cost=sim.CONTINUE_COSTS[outcome_class],
        intervene_realized_cost=sim.INTERVENE_COSTS[outcome_class],
    )


def _generate_pair(group: dict[str, Any], local_index: int,
                   configuration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    partition = group["partition"]
    if partition not in PARTITIONS:
        raise Phase98DataError(f"generation forbidden for partition {partition}")
    if type(local_index) is not int or not 0 <= local_index < 256:
        raise Phase98DataError("scenario index is outside the frozen group")
    seed = group["scenario_seed_start_inclusive"] + local_index
    generated = sim.build_scenario(
        root_seed=group["partition_root_seed"], scenario_seed=seed,
        mechanism=group["mechanism"],
    )
    if generated.policy.scenario_id != generated.evaluator.scenario_id:
        raise Phase98DataError("Phase 9.7 builder returned mismatched scenario identifiers")
    vector = extract_features(generated.policy.snapshot.observation)
    features = vector.as_ordered_dict()
    if tuple(features) != FEATURE_ORDER or set(features) != set(FEATURE_ORDER):
        raise Phase98DataError("observation features differ from the frozen four-feature order")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in features.values()
    ):
        raise Phase98DataError("observation contains an invalid feature value")
    observation = {
        "schema_version": "phase9.8-policy-observation-v1",
        "partition": partition,
        "configuration_group_id": group["configuration_id"],
        "mechanism": group["mechanism"],
        "scenario_id": generated.policy.scenario_id,
        "scenario_seed": seed,
        "root_seed": group["partition_root_seed"],
        "feature_names": list(FEATURE_ORDER),
        "features": features,
    }
    potential = configured_potential_outcomes(generated.policy, generated.evaluator, configuration)
    evaluator_record = {
        "schema_version": "phase9.8-evaluator-outcome-v1",
        "partition": partition,
        "configuration_group_id": group["configuration_id"],
        "mechanism": group["mechanism"],
        "scenario_id": generated.evaluator.scenario_id,
        "continue_passed": potential.continue_passed,
        "intervene_passed": potential.intervene_passed,
        "outcome_class": potential.outcome_class,
        "outcome_class_index": CLASS_INDEX[potential.outcome_class],
        "continue_realized_cost": potential.continue_realized_cost,
        "intervene_realized_cost": potential.intervene_realized_cost,
        "evaluator_only": True,
    }
    return observation, evaluator_record


def _line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_line(row) + "\n" for row in records), encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _feature_matrix(observations: list[dict[str, Any]],
                    feature_names: tuple[str, ...] = FEATURE_ORDER) -> np.ndarray:
    matrix = np.asarray(
        [[row["features"][name] for name in feature_names] for row in observations],
        dtype=np.float64,
    )
    if matrix.shape != (len(observations), len(feature_names)) or not np.isfinite(matrix).all():
        raise Phase98DataError("training feature matrix has invalid shape or values")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise Phase98DataError("training features must lie in [0, 1]")
    return matrix


def _validate_separated_records(observations: list[dict[str, Any]],
                                outcomes: list[dict[str, Any]],
                                expected_count: int, allowed_group_ids: set[str]) -> None:
    if len(observations) != expected_count or len(outcomes) != expected_count:
        raise Phase98DataError("partition record count differs from the frozen manifest")
    obs_ids = [row["scenario_id"] for row in observations]
    out_ids = [row["scenario_id"] for row in outcomes]
    if len(set(obs_ids)) != expected_count or set(obs_ids) != set(out_ids) or len(set(out_ids)) != expected_count:
        raise Phase98DataError("observation/evaluator identifiers are duplicate or mismatched")
    if any(row["configuration_group_id"] not in allowed_group_ids for row in observations + outcomes):
        raise Phase98DataError("partition contains a group outside the frozen manifest")
    if any("outcome_class" in row or "continue_passed" in row or "intervene_passed" in row for row in observations):
        raise Phase98DataError("evaluator-only labels leaked into policy observations")
    if any("features" in row or "failure_prob" in row for row in outcomes):
        raise Phase98DataError("policy observation features leaked into evaluator records")
    for row in observations:
        if row["feature_names"] != list(FEATURE_ORDER) or tuple(row["features"]) != FEATURE_ORDER:
            raise Phase98DataError("feature schema/order mismatch")
    if any(row.get("evaluator_only") is not True or row.get("outcome_class") not in CLASS_INDEX for row in outcomes):
        raise Phase98DataError("evaluator label schema is invalid")
    _feature_matrix(observations)


def generate_authorized_data() -> dict[str, Any]:
    """Generate only fit, calibration-designated, and validation records."""
    configs, manifest, protocol_hash = load_and_validate_design()
    if DATA_ROOT.exists() or MODEL_ROOT.exists():
        raise FileExistsError("refusing to overwrite Phase 9.8 training or model artifacts")
    config_by_id = {row["configuration_id"]: row for row in configs["configurations"]}
    groups_by_partition = {
        part: [group for group in manifest["groups"] if group["partition"] == part]
        for part in PARTITIONS
    }
    all_data: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for partition in PARTITIONS:
        observations: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        for group in groups_by_partition[partition]:
            configuration = config_by_id[group["configuration_id"]]
            for local_index in range(256):
                obs, evaluator = _generate_pair(group, local_index, configuration)
                observations.append(obs)
                outcomes.append(evaluator)
        allowed = {g["configuration_id"] for g in groups_by_partition[partition]}
        _validate_separated_records(
            observations, outcomes, EXPECTED_GROUPS[partition] * 256, allowed,
        )
        all_data[partition] = (observations, outcomes)

    # Determinism audit: replay first and last members of each authorized partition.
    determinism_probes = 0
    for partition in PARTITIONS:
        partition_groups = groups_by_partition[partition]
        for group, local_index in ((partition_groups[0], 0), (partition_groups[-1], 255)):
            config = config_by_id[group["configuration_id"]]
            obs_a, out_a = _generate_pair(group, local_index, config)
            obs_b, out_b = _generate_pair(group, local_index, config)
            if _canonical_bytes([obs_a, out_a]) != _canonical_bytes([obs_b, out_b]):
                raise Phase98DataError("deterministic replay mismatch")
            determinism_probes += 1

    # All records pass validation before creating any output directory.
    DATA_ROOT.mkdir(parents=True, exist_ok=False)
    (DATA_ROOT / "policy_observations").mkdir()
    (DATA_ROOT / "evaluator_outcomes").mkdir()
    count_summary: dict[str, Any] = {}
    data_hashes: dict[str, str] = {}
    for partition, (observations, outcomes) in all_data.items():
        obs_path = DATA_ROOT / "policy_observations" / f"{partition}.jsonl"
        out_path = DATA_ROOT / "evaluator_outcomes" / f"{partition}.jsonl"
        _write_jsonl(obs_path, observations)
        _write_jsonl(out_path, outcomes)
        class_counts = Counter(row["outcome_class"] for row in outcomes)
        count_summary[partition] = {
            "groups": len(groups_by_partition[partition]),
            "scenarios": len(observations),
            "class_counts": {name: class_counts.get(name, 0) for name in EXPECTED_CLASS_NAMES},
            "feature_min": [float(v) for v in _feature_matrix(observations).min(axis=0)],
            "feature_max": [float(v) for v in _feature_matrix(observations).max(axis=0)],
        }
        data_hashes[str(obs_path)] = _file_sha256(obs_path)
        data_hashes[str(out_path)] = _file_sha256(out_path)
    audit = {
        "audit_version": ARTIFACT_VERSION,
        "protocol_sha256": protocol_hash,
        "configuration_table_sha256": EXPECTED_CONFIG_SHA256,
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "authorized_partitions_only": list(PARTITIONS),
        "generated_counts": count_summary,
        "deterministic_replay_probes": determinism_probes,
        "random_stream_names": sorted(sim.STREAM_NAMES),
        "feature_order": list(FEATURE_ORDER),
        "class_order": list(EXPECTED_CLASS_NAMES),
        "separation_checks": "PASS",
        "model_training_performed": False,
        "pilot_or_heldout_generated": False,
        "data_files_sha256": data_hashes,
    }
    audit_path = DATA_ROOT / "generation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return audit


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(EXPECTED_CLASS_NAMES), dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _select_c(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_results or any(result.get("C") not in C_GRID for result in candidate_results):
        raise Phase98DataError("candidate results do not match the frozen C grid")
    if {result["C"] for result in candidate_results} != set(C_GRID):
        raise Phase98DataError("all frozen C candidates are required")
    if any(not math.isfinite(float(result["validation_multiclass_log_loss"]))
           for result in candidate_results):
        raise Phase98DataError("validation log loss must be finite")
    return min(candidate_results, key=lambda result: (result["validation_multiclass_log_loss"], result["C"]))


def _reliability_bins(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    edges = np.linspace(0.0, 1.0, 11)
    for class_index, class_name in enumerate(EXPECTED_CLASS_NAMES):
        bins = []
        for index in range(10):
            lower, upper = float(edges[index]), float(edges[index + 1])
            if index == 9:
                mask = (probabilities[:, class_index] >= lower) & (probabilities[:, class_index] <= upper)
            else:
                mask = (probabilities[:, class_index] >= lower) & (probabilities[:, class_index] < upper)
            count = int(mask.sum())
            bins.append({
                "lower": lower, "upper": upper, "count": count,
                "mean_predicted_probability": float(probabilities[mask, class_index].mean()) if count else None,
                "observed_class_rate": float((y_true[mask] == class_index).mean()) if count else None,
            })
        result[class_name] = bins
    return result


def _load_partition(partition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if partition not in PARTITIONS:
        raise Phase98DataError(f"partition is not authorized for this phase: {partition}")
    obs = _read_jsonl(DATA_ROOT / "policy_observations" / f"{partition}.jsonl")
    outcomes = _read_jsonl(DATA_ROOT / "evaluator_outcomes" / f"{partition}.jsonl")
    return obs, outcomes


def train_and_select() -> dict[str, Any]:
    """Fit exactly the frozen C grid on fit data and select by validation log loss."""
    configs, manifest, protocol_hash = load_and_validate_design()
    if not DATA_ROOT.is_dir() or MODEL_ROOT.exists():
        raise FileNotFoundError("authorized Phase 9.8 training data missing or model output already exists")
    loaded = {partition: _load_partition(partition) for partition in PARTITIONS}
    groups_by_part = {
        part: {g["configuration_id"] for g in manifest["groups"] if g["partition"] == part}
        for part in PARTITIONS
    }
    for partition, (obs, out) in loaded.items():
        _validate_separated_records(obs, out, EXPECTED_GROUPS[partition] * 256, groups_by_part[partition])
    if any(groups_by_part[a] & groups_by_part[b] for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i+1:]):
        raise Phase98DataError("group leakage across authorized partitions")

    fit_obs, fit_out = loaded["fit"]
    val_obs, val_out = loaded["validation"]
    y_fit = np.asarray([CLASS_INDEX[row["outcome_class"]] for row in fit_out], dtype=int)
    y_val = np.asarray([CLASS_INDEX[row["outcome_class"]] for row in val_out], dtype=int)
    if set(y_fit.tolist()) != set(range(4)):
        raise Phase98DataError("fit labels do not contain all four required classes; protocol stop condition")
    x_fit, x_val = _feature_matrix(fit_obs), _feature_matrix(val_obs)
    if x_fit.shape != (10240, 4) or x_val.shape != (2560, 4):
        raise Phase98DataError("fit/validation matrix shape differs from frozen partition counts")

    calibration_obs, calibration_out = loaded["calibration"]
    y_cal = np.asarray([CLASS_INDEX[row["outcome_class"]] for row in calibration_out], dtype=int)
    if MODEL_ROOT.exists():
        raise FileExistsError("refusing to overwrite existing Phase 9.8 model directory")
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    (MODEL_ROOT / "secondary_variants").mkdir()
    selected_models: dict[str, tuple[float, Pipeline, Path, str, list[dict[str, Any]]]] = {}
    all_validation_results: dict[str, Any] = {}
    for variant_name, variant_features in VARIANTS.items():
        x_fit_variant = _feature_matrix(fit_obs, variant_features)
        x_val_variant = _feature_matrix(val_obs, variant_features)
        candidate_results: list[dict[str, Any]] = []
        candidates: dict[float, Pipeline] = {}
        for c_value in C_GRID:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=c_value, solver="lbfgs", max_iter=1000,
                    random_state=manifest["root_seeds"]["fit"],
                )),
            ])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(x_fit_variant, y_fit)
            convergence_warnings = [
                str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
            ]
            if convergence_warnings:
                raise Phase98DataError(f"candidate {variant_name} C={c_value} did not converge")
            classifier = pipeline.named_steps["classifier"]
            if classifier.classes_.tolist() != [0, 1, 2, 3]:
                raise Phase98DataError(f"candidate {variant_name} C={c_value} has unexpected class order")
            probabilities = pipeline.predict_proba(x_val_variant)
            loss = float(log_loss(y_val, probabilities, labels=[0, 1, 2, 3]))
            candidate_results.append({"C": c_value, "validation_multiclass_log_loss": loss})
            candidates[c_value] = pipeline
        # Exact score ties resolve toward lower C; no other validation metric is used.
        selected = _select_c(candidate_results)
        selected_c = float(selected["C"])
        selected_model = candidates[selected_c]
        all_validation_results[variant_name] = {
            "feature_names": list(variant_features),
            "candidates": candidate_results,
            "selected_C": selected_c,
        }
        artifact_path = (
            MODEL_ROOT / "proposed_multinomial_logistic.joblib"
            if variant_name == "full_features"
            else MODEL_ROOT / "secondary_variants" / f"{variant_name}.joblib"
        )
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "variant_name": variant_name,
            "pipeline": selected_model,
            "feature_names": list(variant_features),
            "class_names": list(EXPECTED_CLASS_NAMES),
            "cost_matrix": PHASE97_COST_MATRIX,
            "selected_C": selected_c,
            "classifier_classes": [int(v) for v in selected_model.named_steps["classifier"].classes_],
            "objective": "multinomial LogisticRegression maximum likelihood; frozen cost matrix is applied by expected-cost rule at inference",
        }
        joblib.dump(artifact, artifact_path, compress=3)
        artifact_hash = _file_sha256(artifact_path)
        selected_models[variant_name] = (
            selected_c, selected_model, artifact_path, artifact_hash, candidate_results,
        )

    selected_c, model, artifact_path, model_hash, primary_candidates = selected_models["full_features"]
    x_cal = _feature_matrix(calibration_obs, FEATURE_ORDER)
    calibration_probabilities = model.predict_proba(x_cal)
    calibration_diagnostics = {
        "purpose": "descriptive only; not used to fit, calibrate, select C, or alter decisions",
        "groups": len(groups_by_part["calibration"]),
        "scenarios": len(y_cal),
        "class_counts": {name: int(np.sum(y_cal == i)) for i, name in enumerate(EXPECTED_CLASS_NAMES)},
        "multiclass_brier_score": _multiclass_brier(y_cal, calibration_probabilities),
        "multiclass_log_loss": float(log_loss(y_cal, calibration_probabilities, labels=[0, 1, 2, 3])),
        "reliability_bins_one_vs_rest": _reliability_bins(y_cal, calibration_probabilities),
    }

    data_hashes = {
        str(path): _file_sha256(path)
        for path in sorted(DATA_ROOT.rglob("*.jsonl"))
    }
    training_group_ids = sorted(groups_by_part["fit"])
    model_hashes = {
        name: {"path": str(path), "sha256": digest, "selected_C": c_value,
               "feature_names": list(VARIANTS[name])}
        for name, (c_value, _, path, digest, _) in selected_models.items()
    }
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_file": str(artifact_path),
        "artifact_sha256": model_hash,
        "protocol_file": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "configuration_table_sha256": EXPECTED_CONFIG_SHA256,
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "feature_names": list(FEATURE_ORDER),
        "class_names": list(EXPECTED_CLASS_NAMES),
        "cost_matrix": PHASE97_COST_MATRIX,
        "classifier_classes": [int(v) for v in model.named_steps["classifier"].classes_],
        "training_group_ids": training_group_ids,
        "training_group_count": len(training_group_ids),
        "training_scenario_count": len(fit_obs),
        "training_class_counts": {name: int(np.sum(y_fit == i)) for i, name in enumerate(EXPECTED_CLASS_NAMES)},
        "calibration_group_ids": sorted(groups_by_part["calibration"]),
        "calibration_scenario_count": len(calibration_obs),
        "validation_group_ids": sorted(groups_by_part["validation"]),
        "validation_scenario_count": len(val_obs),
        "partition_root_seeds": {name: manifest["root_seeds"][name] for name in PARTITIONS},
        "training_model_seed": manifest["root_seeds"]["fit"],
        "scenario_seed_contract": manifest["scenario_seed_allocation"],
        "selected_C": selected_c,
        "selection_rule": "minimum validation multiclass log loss among C=[0.1,1,10]; exact ties resolve to smaller C",
        "validation_candidate_results": all_validation_results,
        "model_variants": model_hashes,
        "estimator_parameters": model.named_steps["classifier"].get_params(deep=False),
        "pipeline_steps": [type(step).__name__ for _, step in model.steps],
        "multinomial_semantics": "lbfgs multinomial multiclass behavior; installed scikit-learn 1.9.1 no longer exposes multi_class parameter",
        "calibration_diagnostics": calibration_diagnostics,
        "calibration_diagnostics_used_for_selection": False,
        "data_file_sha256": data_hashes,
        "training_data_generation": "offline Phase 9.7 observations plus Phase 9.8 evaluator-only configured potential outcomes",
        "pilot_or_heldout_generated": False,
        "phase9_2_artifact_reused": False,
    }
    metadata_path = MODEL_ROOT / "model_metadata.json"
    metadata["metadata_sha256"] = _sha256_bytes(_canonical_bytes(metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    # Read-only verification of the newly written, locally trained artifact.
    for variant_name, (c_value, _, variant_path, variant_hash, _) in selected_models.items():
        loaded_artifact = joblib.load(variant_path)
        expected_features = list(VARIANTS[variant_name])
        if (loaded_artifact.get("artifact_version") != ARTIFACT_VERSION
                or loaded_artifact.get("variant_name") != variant_name
                or loaded_artifact.get("feature_names") != expected_features
                or loaded_artifact.get("class_names") != list(EXPECTED_CLASS_NAMES)
                or loaded_artifact.get("selected_C") != c_value
                or loaded_artifact["pipeline"].named_steps["classifier"].classes_.tolist() != [0, 1, 2, 3]
                or _file_sha256(variant_path) != variant_hash):
            raise Phase98DataError(f"saved model artifact verification failed: {variant_name}")
        probe_x = _feature_matrix(val_obs[:3], VARIANTS[variant_name])
        probe = loaded_artifact["pipeline"].predict_proba(probe_x)
        if probe.shape != (3, 4) or not np.isfinite(probe).all() or not np.allclose(probe.sum(axis=1), 1.0):
            raise Phase98DataError(f"saved model probability probe failed: {variant_name}")
    metadata["verification"] = {
        "joblib_reload": "PASS",
        "feature_order": "PASS",
        "class_order": "PASS",
        "probability_shape_and_simplex": "PASS",
        "artifact_sha256_rechecked": _file_sha256(artifact_path) == model_hash,
        "verified_probe_rows_per_model": 3,
        "verified_model_count": len(selected_models),
    }
    # Refresh metadata hash after adding verification result.
    metadata.pop("metadata_sha256")
    metadata["metadata_sha256"] = _sha256_bytes(_canonical_bytes(metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report = {
        "status": "training_and_selection_complete",
        "selected_C": selected_c,
        "validation_candidate_results": all_validation_results,
        "secondary_model_hashes": {
            name: {"path": str(path), "sha256": digest, "selected_C": c_value}
            for name, (c_value, _, path, digest, _) in selected_models.items()
            if name != "full_features"
        },
        "fit_class_counts": metadata["training_class_counts"],
        "calibration_class_counts": calibration_diagnostics["class_counts"],
        "validation_class_counts": {name: int(np.sum(y_val == i)) for i, name in enumerate(EXPECTED_CLASS_NAMES)},
        "calibration_diagnostics": calibration_diagnostics,
        "artifact_path": str(artifact_path),
        "artifact_sha256": model_hash,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata["metadata_sha256"],
        "pilot_or_heldout_generated": False,
    }
    (MODEL_ROOT / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("generate", "train"))
    args = parser.parse_args()
    try:
        if args.mode == "generate":
            print(json.dumps(generate_authorized_data(), indent=2, sort_keys=True))
        else:
            print(json.dumps(train_and_select(), indent=2, sort_keys=True))
        return 0
    except (FrozenDesignError, Phase98DataError, FileExistsError, FileNotFoundError, ValueError) as exc:
        # No exception detail from external services is involved; all errors are
        # local schema/IO/training diagnostics. Keep output bounded and explicit.
        print(f"phase9.8 {args.mode} failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
