"""Versioned Phase 9.8 model adapter with a fail-closed pilot lock gate.

This module is separate from the unchanged Phase 9.7 artifact gate.  The
normal loader refuses draft/unapproved locks before it can deserialize a model.
Tests may inject a fake loader to exercise validation without reading joblib.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES, PHASE97_COST_MATRIX, expected_costs


ADAPTER_VERSION = "phase9-8-pilot-adapter-v1"
MODEL_VERSION = "phase9.8-multinomial-cost-sensitive-v1"
LOCK_PATH = Path("results/phase9_8_pilot_lock.json")
MODEL_PATH = Path("models/phase9_8_final/proposed_multinomial_logistic.joblib")
METADATA_PATH = Path("models/phase9_8_final/model_metadata.json")
TRAINING_REPORT_PATH = Path("models/phase9_8_final/training_report.json")
PILOT_PREREG_PATH = Path("results/phase9_8_pilot_preregistration.md")
PROTOCOL_PATH = Path("results/phase9_8_protocol.md")
CONFIGURATION_PATH = Path("results/phase9_8_configuration_table.json")
SPLIT_PATH = Path("results/phase9_8_split_manifest.json")
EXPECTED_FEATURE_NAMES = tuple(FEATURE_ORDER)
EXPECTED_CLASS_NAMES = tuple(OUTCOME_CLASSES)
EXPECTED_C = 1.0
EXPECTED_PROTOCOL_SHA256 = "2858917ce9b4ac2bb25289d7f16cb5d71d467b95c1d59598a36ee034380e453e"
EXPECTED_CONFIGURATION_SHA256 = "efc4caf9cf570038a772bb76df57215cc412e3fce0e207566119db6dd83a3963"
EXPECTED_SPLIT_SHA256 = "9686e7d0a53b5ec261afd9f02cca4d3c80bd036e4d6a55399aaf19848c1eda99"
_LOAD_GATE_TOKEN = object()


class Phase98ArtifactError(ValueError):
    """A Phase 9.8 lock, metadata, or model contract failed validation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_content_hash(metadata: Mapping[str, Any]) -> str:
    payload = dict(metadata)
    payload.pop("metadata_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_metadata(metadata: Any) -> None:
    """Validate frozen metadata without deserializing the fitted artifact."""
    if not isinstance(metadata, dict):
        raise Phase98ArtifactError("Phase 9.8 metadata must be a JSON object")
    exact = {
        "artifact_version": MODEL_VERSION,
        "feature_names": list(EXPECTED_FEATURE_NAMES),
        "class_names": list(EXPECTED_CLASS_NAMES),
        "classifier_classes": [0, 1, 2, 3],
        "cost_matrix": PHASE97_COST_MATRIX,
        "pipeline_steps": ["StandardScaler", "LogisticRegression"],
        "selected_C": EXPECTED_C,
        "selection_rule": "minimum validation multiclass log loss among C=[0.1,1,10]; exact ties resolve to smaller C",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "configuration_table_sha256": EXPECTED_CONFIGURATION_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "phase9_2_artifact_reused": False,
        "pilot_or_heldout_generated": False,
    }
    if any(metadata.get(key) != value for key, value in exact.items()):
        raise Phase98ArtifactError("Phase 9.8 metadata does not match its frozen contract")
    if metadata.get("model_variants", {}).get("full_features", {}).get("selected_C") != EXPECTED_C:
        raise Phase98ArtifactError("metadata full-feature variant or selected C is invalid")
    if metadata.get("artifact_file", "").replace("/", "\\") != str(MODEL_PATH).replace("/", "\\"):
        raise Phase98ArtifactError("metadata model path is not the frozen full-feature artifact")
    if metadata.get("model_variants", {}).get("full_features", {}).get("sha256") != metadata.get("artifact_sha256"):
        raise Phase98ArtifactError("full-feature model variant hash differs from artifact hash")
    if metadata.get("model_variants", {}).get("full_features", {}).get("feature_names") != list(EXPECTED_FEATURE_NAMES):
        raise Phase98ArtifactError("metadata full-feature order is invalid")
    params = metadata.get("estimator_parameters")
    if not isinstance(params, dict) or params.get("C") != EXPECTED_C or params.get("solver") != "lbfgs":
        raise Phase98ArtifactError("saved estimator parameters differ from the selected model")
    validation = metadata.get("validation_candidate_results", {}).get("full_features", {})
    if validation.get("selected_C") != EXPECTED_C:
        raise Phase98ArtifactError("validation selection metadata is missing or inconsistent")
    candidates = validation.get("candidates")
    if not isinstance(candidates, list) or {row.get("C") for row in candidates} != {0.1, 1.0, 10.0}:
        raise Phase98ArtifactError("frozen C selection candidates are incomplete")
    try:
        selected = min(candidates, key=lambda row: (float(row["validation_multiclass_log_loss"]), float(row["C"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase98ArtifactError("validation candidate losses are invalid") from exc
    if float(selected["C"]) != EXPECTED_C:
        raise Phase98ArtifactError("selected C does not follow frozen validation log-loss rule")
    if metadata.get("training_group_count") != 40 or len(metadata.get("training_group_ids", [])) != 40:
        raise Phase98ArtifactError("model was not trained on the frozen fit-group count")
    if not metadata.get("metadata_sha256") or metadata["metadata_sha256"] != _metadata_content_hash(metadata):
        raise Phase98ArtifactError("metadata self-hash is invalid")


def _expected_hashes(root: Path, metadata: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "protocol": PROTOCOL_PATH,
        "configuration_table": CONFIGURATION_PATH,
        "split_manifest": SPLIT_PATH,
        "model": MODEL_PATH,
        "metadata": METADATA_PATH,
        "training_report": TRAINING_REPORT_PATH,
        "pilot_preregistration": PILOT_PREREG_PATH,
    }
    for key, path in paths.items():
        if not (root / path).is_file():
            raise Phase98ArtifactError(f"required frozen Phase 9.8 file is missing: {key}")
    return {key: _sha256(root / path) for key, path in paths.items()}


def _validate_model(model: Any) -> None:
    """Validate the deserialized pipeline's public fitted interface."""
    try:
        steps = list(model.named_steps.items())
        names = [name for name, _ in steps]
        classifier = model.named_steps["classifier"]
        classes = np.asarray(classifier.classes_).tolist()
        feature_count = int(model.n_features_in_)
        c_value = float(classifier.C)
    except Exception as exc:
        raise Phase98ArtifactError("deserialized model has an invalid pipeline interface") from exc
    if names != ["scaler", "classifier"] or feature_count != 4:
        raise Phase98ArtifactError("deserialized pipeline steps or feature count are invalid")
    if type(steps[0][1]).__name__ != "StandardScaler" or type(classifier).__name__ != "LogisticRegression":
        raise Phase98ArtifactError("deserialized pipeline component types are invalid")
    if classes != [0, 1, 2, 3]:
        raise Phase98ArtifactError("deserialized classifier class order is invalid")
    if c_value != EXPECTED_C:
        raise Phase98ArtifactError("deserialized classifier C differs from frozen selection")
    if isinstance(getattr(steps[0][1], "n_features_in_", None), bool) or getattr(steps[0][1], "n_features_in_", None) != 4:
        raise Phase98ArtifactError("deserialized scaler feature count is invalid")


class Phase98ProposedPolicyAdapter:
    """Single-prediction Phase 9.8 adapter; no branching or rollback."""

    def __init__(self, model: Any, metadata: Mapping[str, Any], *, artifact_sha256: str,
                 _gate_token: object | None = None):
        if _gate_token is not _LOAD_GATE_TOKEN:
            raise Phase98ArtifactError("construct the adapter only through the lock-gated loader")
        validate_metadata(dict(metadata))
        _validate_model(model)
        if artifact_sha256 != metadata.get("artifact_sha256"):
            raise Phase98ArtifactError("loaded model hash does not match metadata")
        self._model = model
        self.artifact_sha256 = artifact_sha256

    def decide(self, features: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(features, Mapping) or tuple(features) != EXPECTED_FEATURE_NAMES:
            raise ValueError("Phase 9.8 inference requires the exact ordered feature schema")
        values = []
        for name in EXPECTED_FEATURE_NAMES:
            value = features[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"feature {name} must be a numeric scalar")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"feature {name} must be finite and within [0, 1]")
            values.append(value)
        raw = np.asarray(self._model.predict_proba(np.asarray([values], dtype=np.float64)), dtype=object)
        if raw.shape != (1, 4):
            raise ValueError("Phase 9.8 classifier returned an invalid probability shape")
        probabilities: dict[str, float] = {}
        for name, value in zip(EXPECTED_CLASS_NAMES, raw[0], strict=True):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError("Phase 9.8 classifier returned a nonnumeric probability")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("Phase 9.8 classifier returned an invalid probability")
            probabilities[name] = number
        diagnostics = expected_costs(probabilities)
        return {
            "adapter_version": ADAPTER_VERSION,
            "class_probabilities": probabilities,
            "expected_continue_cost": diagnostics["expected_continue_cost"],
            "expected_intervene_cost": diagnostics["expected_intervene_cost"],
            "selected_action": diagnostics["selected_action"],
        }


def load_phase98_adapter(
    *, root: Path | str = Path("."), lock_path: Path | str = LOCK_PATH,
    approved_lock_sha256: str | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> Phase98ProposedPolicyAdapter:
    """Fail closed unless a separately approved lock hash is explicitly supplied.

    ``model_loader`` is an injection point for offline contract tests. Production
    callers should omit it; the default uses joblib only after all gates pass.
    """
    project_root = Path(root)
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = project_root / lock_file
    if not lock_file.is_file():
        raise Phase98ArtifactError("pilot lock is missing")
    lock_raw_hash = _sha256(lock_file)
    if not approved_lock_sha256 or lock_raw_hash != approved_lock_sha256:
        raise Phase98ArtifactError("pilot lock has not been independently approved")
    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase98ArtifactError("pilot lock is unreadable or malformed") from exc
    if (lock.get("schema_version") != "phase9-8-independent-pilot-lock-v1"
            or lock.get("approval_status") != "APPROVED"
            or lock.get("adapter_version") != ADAPTER_VERSION):
        raise Phase98ArtifactError("pilot lock is not approved for this adapter version")
    lock_payload = dict(lock)
    recorded_lock_hash = lock_payload.pop("lock_sha256", None)
    if recorded_lock_hash != hashlib.sha256(_canonical_bytes(lock_payload)).hexdigest():
        raise Phase98ArtifactError("pilot lock content hash is invalid")
    hashes = lock.get("sha256")
    if not isinstance(hashes, dict):
        raise Phase98ArtifactError("pilot lock hash table is invalid")
    actual_hashes = _expected_hashes(project_root, {})
    if hashes != actual_hashes:
        raise Phase98ArtifactError("frozen source or model artifact hash differs from approved lock")
    try:
        config_table = json.loads((project_root / CONFIGURATION_PATH).read_text(encoding="utf-8"))
        split_manifest = json.loads((project_root / SPLIT_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase98ArtifactError("frozen configuration or split manifest is unreadable") from exc
    canonical_hashes = {
        "configuration_table_sha256": EXPECTED_CONFIGURATION_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
    }
    if (config_table.get("configuration_table_sha256") != EXPECTED_CONFIGURATION_SHA256
            or split_manifest.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or split_manifest.get("configuration_table_sha256") != EXPECTED_CONFIGURATION_SHA256
            or {key: lock.get("canonical_design_hashes", {}).get(key) for key in canonical_hashes} != canonical_hashes):
        raise Phase98ArtifactError("lock canonical design hashes do not match frozen artifacts")
    pilot_groups = [group for group in split_manifest.get("groups", []) if group.get("partition") == "pilot"]
    expected_group_ids = [group["configuration_id"] for group in pilot_groups]
    expected_scenarios = sum(group.get("scenario_count", 0) for group in pilot_groups)
    expected_pilot = {
        "root_seed": 20269804,
        "configuration_group_ids": expected_group_ids,
        "scenario_count": expected_scenarios,
    }
    if (len(pilot_groups) != 10 or expected_scenarios != 2560
            or lock.get("pilot_partition") != expected_pilot):
        raise Phase98ArtifactError("pilot lock partition assignment differs from the frozen split")
    metadata_path = project_root / METADATA_PATH
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase98ArtifactError("model metadata is unreadable or malformed") from exc
    validate_metadata(metadata)
    if metadata.get("artifact_sha256") != actual_hashes["model"]:
        raise Phase98ArtifactError("model digest differs from sidecar metadata")
    if metadata.get("metadata_sha256") != lock.get("metadata_content_sha256"):
        raise Phase98ArtifactError("model metadata content hash differs from approved lock")
    if lock.get("canonical_design_hashes", {}).get("metadata_content_sha256") != metadata.get("metadata_sha256"):
        raise Phase98ArtifactError("lock does not bind the canonical metadata hash")
    if lock.get("selected_C") != EXPECTED_C or lock.get("feature_order") != list(EXPECTED_FEATURE_NAMES) or lock.get("class_order") != list(EXPECTED_CLASS_NAMES):
        raise Phase98ArtifactError("pilot lock schema or selected hyperparameter is invalid")
    if model_loader is None:
        import joblib
        model_loader = joblib.load
    model = model_loader(project_root / MODEL_PATH)
    return Phase98ProposedPolicyAdapter(
        model, metadata, artifact_sha256=actual_hashes["model"], _gate_token=_LOAD_GATE_TOKEN,
    )
