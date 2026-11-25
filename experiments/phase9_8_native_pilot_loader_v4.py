"""Fail-closed v4 loader for the wrapped Phase 9.8 fitted-model artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import (
    EXPECTED_C, METADATA_PATH, MODEL_PATH, PHASE97_COST_MATRIX,
    Phase98ArtifactError, Phase98ProposedPolicyAdapter, _LOAD_GATE_TOKEN,
    _canonical_bytes, _metadata_content_hash, _sha256, validate_metadata,
)
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design
from experiments.phase9_8_native_pilot_loader_v2 import EXPECTED_POLICY_SET, _read_json
from experiments.phase9_8_native_pilot_loader_v3 import (
    NATIVE_PILOT_LOCK_V3_PATH, NATIVE_PILOT_LOADER_V3_VERSION,
    EXPECTED_LOCK_SCHEMA_V3, EXPECTED_V2_LOCK_RAW_SHA256,
    EXPECTED_V2_LOCK_CANONICAL_SHA256, SUPERSEDING_PREREGISTRATION_V3_PATH,
    _BOUND_PATHS_V3, _actual_hashes as _actual_hashes_v3,
    _validate_v3_lock,
)


NATIVE_PILOT_LOADER_V4_VERSION = "phase9-8-native-pilot-loader-v4-wrapper"
NATIVE_PILOT_LOCK_V4_PATH = Path("results/phase9_8_native_pilot_lock_v4.json")
SUPERSEDING_DOCUMENT_V4_PATH = Path("results/phase9_8_pilot_wrapper_loader_addendum_v4.md")
EXPECTED_V3_LOCK_RAW_SHA256 = "5febea673975f375fc7e758134bcd1a67f22fe564cec72561220a4da195fde18"
EXPECTED_V3_LOCK_CANONICAL_SHA256 = "e9a8c516b04fa302ed457c06608b904700d91ea10fd008746d5ef236a2dfd079"
EXPECTED_LOCK_SCHEMA_V4 = "phase9-8-native-pilot-independent-lock-v4"
WRAPPER_KEYS = {
    "artifact_version", "variant_name", "pipeline", "feature_names", "class_names",
    "cost_matrix", "selected_C", "classifier_classes", "objective",
}
EXPECTED_OBJECTIVE = (
    "multinomial LogisticRegression maximum likelihood; frozen cost matrix is applied "
    "by expected-cost rule at inference"
)

_BOUND_PATHS_V4 = dict(_BOUND_PATHS_V3)
_BOUND_PATHS_V4.update({
    "predecessor_v3_lock": NATIVE_PILOT_LOCK_V3_PATH,
    "superseding_wrapper_addendum_v4": SUPERSEDING_DOCUMENT_V4_PATH,
    "native_loader_v4": Path("experiments/phase9_8_native_pilot_loader_v4.py"),
    "lock_builder_v4": Path("scripts/freeze_phase9_8_native_pilot_lock_v4.py"),
    "loader_v4_tests": Path("tests/test_phase9_8_native_pilot_loader_v4.py"),
    "orchestrator_v4": Path("experiments/phase9_8_pilot_orchestrator_v4.py"),
    "entrypoint_v4": Path("experiments/phase9_8_native_pilot_v4.py"),
})


class NativePilotV4ArtifactError(Phase98ArtifactError):
    """The v4 lock or wrapped model artifact failed its frozen contract."""


def _canonical_integrity(document: dict[str, Any], field: str, label: str) -> str:
    payload = dict(document)
    recorded = payload.pop(field, None)
    actual = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if not isinstance(recorded, str) or recorded != actual:
        raise NativePilotV4ArtifactError(f"{label} canonical digest is invalid")
    return actual


def _actual_hashes_v4(root: Path) -> dict[str, str]:
    inherited = _actual_hashes_v3(root)
    output = dict(inherited)
    for name, relative in _BOUND_PATHS_V4.items():
        if name in output:
            continue
        path = root / relative
        if not path.is_file():
            raise NativePilotV4ArtifactError(f"bound file is missing: {name}")
        output[name] = _sha256(path)
    return output


def _validate_v4_lock(
    lock: dict[str, Any], raw_hash: str, expected_raw_hash: str, root: Path,
) -> dict[str, Any]:
    if raw_hash != expected_raw_hash:
        raise NativePilotV4ArtifactError("raw v4 lock digest differs from independent digest")
    if (lock.get("schema_version") != EXPECTED_LOCK_SCHEMA_V4
            or lock.get("approval_status") != "APPROVED"
            or lock.get("loader_version") != NATIVE_PILOT_LOADER_V4_VERSION
            or lock.get("superseding_document") != str(SUPERSEDING_DOCUMENT_V4_PATH).replace("\\", "/")):
        raise NativePilotV4ArtifactError("v4 lock is draft or uses an unsupported contract")
    _canonical_integrity(lock, "lock_canonical_sha256", "v4 lock")
    exact = {
        "comparator_id": "phase9_8_native_prefix_forecast_v1",
        "comparative_policy_set": EXPECTED_POLICY_SET,
        "pilot_test_policy": {
            "in_comparative_set": False,
            "outcome_based_selection_after_freeze": True,
            "comparative_metric_eligible": False,
        },
        "frozen_feature_order": list(FEATURE_ORDER),
        "class_order": list(OUTCOME_CLASSES),
        "selected_C": EXPECTED_C,
        "forecast_count_per_action": 4,
        "forecast_count_total": 8,
        "expected_cost_rule": {
            "continue": "3 * (1 - p_pass_continue)",
            "intervene": "2 + 3 * (1 - p_pass_intervene)",
            "tie": "continue",
        },
        "execution_semantics": "one_decision_counterfactual_simulation",
        "artifact_wrapper_format": "phase9.8-training-wrapper-v1",
        "pilot_scope": {
            "partition": "pilot", "configuration_group_count": 10,
            "scenario_count": 2560, "root_seed": 20269804,
        },
    }
    if any(lock.get(key) != value for key, value in exact.items()):
        raise NativePilotV4ArtifactError("v4 lock protocol fields differ from frozen contract")
    if lock.get("predecessor_v3_lock_raw_sha256") != EXPECTED_V3_LOCK_RAW_SHA256:
        raise NativePilotV4ArtifactError("v4 lock does not bind the approved v3 predecessor")
    if lock.get("predecessor_v3_lock_canonical_sha256") != EXPECTED_V3_LOCK_CANONICAL_SHA256:
        raise NativePilotV4ArtifactError("v4 lock predecessor canonical hash differs")

    recorded_hashes = lock.get("bound_file_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(_BOUND_PATHS_V4):
        raise NativePilotV4ArtifactError("v4 bound-file digest schema is incomplete")
    actual_hashes = _actual_hashes_v4(root)
    if actual_hashes != recorded_hashes:
        raise NativePilotV4ArtifactError("a v4 bound-file digest does not match")

    v3_path = root / NATIVE_PILOT_LOCK_V3_PATH
    v3 = _read_json(v3_path, "approved v3 predecessor lock")
    v3_raw = _sha256(v3_path)
    if v3_raw != EXPECTED_V3_LOCK_RAW_SHA256:
        raise NativePilotV4ArtifactError("approved v3 predecessor raw hash changed")
    if _canonical_integrity(v3, "lock_canonical_sha256", "v3 predecessor") != EXPECTED_V3_LOCK_CANONICAL_SHA256:
        raise NativePilotV4ArtifactError("approved v3 predecessor canonical hash changed")
    if v3.get("approval_status") != "APPROVED":
        raise NativePilotV4ArtifactError("v3 predecessor is not approved")
    _validate_v3_lock(v3, v3_raw, EXPECTED_V3_LOCK_RAW_SHA256, root)

    if lock.get("inherited_frozen_raw_hashes") != v3.get("inherited_frozen_raw_hashes"):
        raise NativePilotV4ArtifactError("v4 inherited frozen hashes differ from approved v3 lock")
    metadata = _read_json(root / METADATA_PATH, "model metadata")
    try:
        validate_metadata(metadata)
    except Phase98ArtifactError as exc:
        raise NativePilotV4ArtifactError("Phase 9.8 model metadata contract failed") from exc
    if (metadata.get("metadata_sha256") != _metadata_content_hash(metadata)
            or metadata.get("metadata_sha256") != lock.get("metadata_content_sha256")
            or metadata.get("artifact_sha256") != actual_hashes["model"]):
        raise NativePilotV4ArtifactError("model, metadata, or model-content binding differs")
    try:
        design = load_frozen_pilot_design(root)
    except (OSError, ValueError) as exc:
        raise NativePilotV4ArtifactError("frozen pilot partition validation failed") from exc
    if (len(design.scenario_specs) != 2560
            or len({spec[0] for spec in design.scenario_specs.values()}) != 10
            or {spec[2] for spec in design.scenario_specs.values()} != {20269804}):
        raise NativePilotV4ArtifactError("pilot partition differs from frozen 10-group design")
    return metadata


def _extract_validated_pipeline(artifact: Any, metadata: dict[str, Any]) -> Pipeline:
    """Validate the training wrapper and return only its already-fitted pipeline."""
    if not isinstance(artifact, dict) or set(artifact) != WRAPPER_KEYS:
        raise NativePilotV4ArtifactError("saved artifact wrapper fields are invalid")
    exact_fields = {
        "artifact_version": metadata.get("artifact_version"),
        "variant_name": "full_features",
        "feature_names": list(FEATURE_ORDER),
        "class_names": list(OUTCOME_CLASSES),
        "cost_matrix": PHASE97_COST_MATRIX,
        "classifier_classes": [0, 1, 2, 3],
        "objective": EXPECTED_OBJECTIVE,
    }
    for name, expected in exact_fields.items():
        if artifact.get(name) != expected:
            raise NativePilotV4ArtifactError(f"saved artifact wrapper field is invalid: {name}")
    selected_c = artifact.get("selected_C")
    if (isinstance(selected_c, bool) or not isinstance(selected_c, (int, float))
            or not np.isfinite(selected_c) or float(selected_c) != EXPECTED_C
            or float(selected_c) != float(metadata.get("selected_C"))):
        raise NativePilotV4ArtifactError("saved artifact selected C differs from metadata")

    pipeline = artifact.get("pipeline")
    if not isinstance(pipeline, Pipeline):
        raise NativePilotV4ArtifactError("saved artifact pipeline has an unexpected type")
    steps = pipeline.named_steps
    if list(steps) != ["scaler", "classifier"]:
        raise NativePilotV4ArtifactError("saved artifact pipeline steps are invalid")
    scaler, classifier = steps["scaler"], steps["classifier"]
    if not isinstance(scaler, StandardScaler) or not isinstance(classifier, LogisticRegression):
        raise NativePilotV4ArtifactError("saved artifact pipeline component types are invalid")

    if (getattr(pipeline, "n_features_in_", None) != len(FEATURE_ORDER)
            or getattr(scaler, "n_features_in_", None) != len(FEATURE_ORDER)
            or getattr(classifier, "n_features_in_", None) != len(FEATURE_ORDER)):
        raise NativePilotV4ArtifactError("saved artifact fitted feature counts are invalid")
    classes = np.asarray(getattr(classifier, "classes_", [])).tolist()
    if classes != [0, 1, 2, 3]:
        raise NativePilotV4ArtifactError("saved artifact fitted class order is invalid")
    fitted_c = getattr(classifier, "C", None)
    if (isinstance(fitted_c, bool) or not isinstance(fitted_c, (int, float))
            or not np.isfinite(fitted_c) or float(fitted_c) != EXPECTED_C):
        raise NativePilotV4ArtifactError("saved artifact fitted classifier C is invalid")

    arrays = (
        (getattr(scaler, "mean_", None), (4,)),
        (getattr(scaler, "scale_", None), (4,)),
        (getattr(scaler, "var_", None), (4,)),
        (getattr(classifier, "coef_", None), (4, 4)),
        (getattr(classifier, "intercept_", None), (4,)),
        (getattr(classifier, "n_iter_", None), (1,)),
    )
    for value, shape in arrays:
        if value is None:
            raise NativePilotV4ArtifactError("saved artifact is missing fitted pipeline state")
        array = np.asarray(value)
        if array.shape != shape or not np.isfinite(array).all():
            raise NativePilotV4ArtifactError("saved artifact fitted pipeline state is invalid")
    samples_seen = getattr(scaler, "n_samples_seen_", None)
    if samples_seen is None:
        raise NativePilotV4ArtifactError("saved scaler is missing fitted sample state")
    sample_array = np.asarray(samples_seen)
    if (sample_array.size == 0 or not np.issubdtype(sample_array.dtype, np.number)
            or not np.isfinite(sample_array).all() or (sample_array <= 0).any()):
        raise NativePilotV4ArtifactError("saved scaler fitted sample state is invalid")
    return pipeline


def load_native_pilot_adapter_v4(
    *, root: Path | str = Path("."),
    lock_path: Path | str = NATIVE_PILOT_LOCK_V4_PATH,
    approved_lock_sha256: str | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> Phase98ProposedPolicyAdapter:
    """Validate the new lock, then unwrap the trusted fitted pipeline."""
    project_root = Path(root)
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = project_root / lock_file
    if not lock_file.is_file() or not isinstance(approved_lock_sha256, str):
        raise NativePilotV4ArtifactError("v4 lock and independent raw digest are required")
    raw_hash = _sha256(lock_file)
    if raw_hash != approved_lock_sha256:
        raise NativePilotV4ArtifactError("raw v4 lock digest differs from independent value")
    lock = _read_json(lock_file, "v4 lock")
    metadata = _validate_v4_lock(lock, raw_hash, approved_lock_sha256, project_root)
    if model_loader is None:
        import joblib
        model_loader = joblib.load
    artifact = model_loader(project_root / MODEL_PATH)
    pipeline = _extract_validated_pipeline(artifact, metadata)
    return Phase98ProposedPolicyAdapter(
        pipeline, metadata, artifact_sha256=lock["bound_file_sha256"]["model"],
        _gate_token=_LOAD_GATE_TOKEN,
    )
