"""Fail-closed loader for the v3 superseding Phase 9.8 preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import (
    EXPECTED_C, METADATA_PATH, MODEL_PATH, Phase98ArtifactError,
    Phase98ProposedPolicyAdapter, _LOAD_GATE_TOKEN, _canonical_bytes,
    _metadata_content_hash, _sha256, validate_metadata,
)
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design
from experiments.phase9_8_native_pilot_loader_v2 import (
    EXPECTED_ORIGINAL_LOCK_RAW_SHA256, EXPECTED_POLICY_SET,
    NATIVE_PILOT_LOCK_V2_PATH, _BOUND_PATHS as V2_BOUND_PATHS,
    _read_json,
)


NATIVE_PILOT_LOADER_V3_VERSION = "phase9-8-native-pilot-loader-v3"
NATIVE_PILOT_LOCK_V3_PATH = Path("results/phase9_8_native_pilot_lock_v3.json")
SUPERSEDING_PREREGISTRATION_V3_PATH = Path("results/phase9_8_pilot_preregistration_supersession_v3.md")
ORIGINAL_LOCK_PATH = Path("results/phase9_8_pilot_lock.json")
NATIVE_V1_LOCK_PATH = Path("results/phase9_8_native_prefix_pilot_lock_v1.json")
EXPECTED_V2_LOCK_RAW_SHA256 = "17e920e01116c9ea62e89c7741af8aa9a7bcecb26fcbbecfbfb88d57c52d3f13"
EXPECTED_V2_LOCK_CANONICAL_SHA256 = "eb201b25d7f11a075777a0889d60be5e1cce962278e6f72a5219a7bf93cf5912"
EXPECTED_LOCK_SCHEMA_V3 = "phase9-8-native-pilot-independent-lock-v3"

_BOUND_PATHS_V3 = dict(V2_BOUND_PATHS)
_BOUND_PATHS_V3.update({
    "predecessor_v2_lock": NATIVE_PILOT_LOCK_V2_PATH,
    "superseding_preregistration_v3": SUPERSEDING_PREREGISTRATION_V3_PATH,
    "native_loader_v3": Path("experiments/phase9_8_native_pilot_loader_v3.py"),
    "lock_builder_v3": Path("scripts/freeze_phase9_8_native_pilot_lock_v3.py"),
    "loader_v3_tests": Path("tests/test_phase9_8_native_pilot_loader_v3.py"),
})


class NativePilotV3ArtifactError(Phase98ArtifactError):
    """The v3 approval lock or its frozen bindings failed validation."""


def _canonical_integrity(document: dict[str, Any], field: str, label: str) -> str:
    payload = dict(document)
    recorded = payload.pop(field, None)
    actual = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if not isinstance(recorded, str) or recorded != actual:
        raise NativePilotV3ArtifactError(f"{label} canonical digest is invalid")
    return actual


def _actual_hashes(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, relative in _BOUND_PATHS_V3.items():
        path = root / relative
        if not path.is_file():
            raise NativePilotV3ArtifactError(f"bound file is missing: {name}")
        output[name] = _sha256(path)
    return output


def _validate_v3_lock(lock: dict[str, Any], raw_hash: str,
                      expected_raw_hash: str, root: Path) -> dict[str, Any]:
    if raw_hash != expected_raw_hash:
        raise NativePilotV3ArtifactError("raw v3 lock digest differs from independently supplied digest")
    if (lock.get("schema_version") != EXPECTED_LOCK_SCHEMA_V3
            or lock.get("approval_status") != "APPROVED"
            or lock.get("loader_version") != NATIVE_PILOT_LOADER_V3_VERSION
            or lock.get("superseding_document") != str(SUPERSEDING_PREREGISTRATION_V3_PATH).replace("\\", "/")):
        raise NativePilotV3ArtifactError("v3 lock is draft or has an unsupported supersession contract")
    _canonical_integrity(lock, "lock_canonical_sha256", "v3 lock")
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
        "pilot_scope": {
            "partition": "pilot", "configuration_group_count": 10,
            "scenario_count": 2560, "root_seed": 20269804,
        },
        "execution_semantics": "one_decision_counterfactual_simulation",
    }
    if any(lock.get(key) != value for key, value in exact.items()):
        raise NativePilotV3ArtifactError("v3 lock protocol fields differ from the supersession document")
    recorded = lock.get("bound_file_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(_BOUND_PATHS_V3):
        raise NativePilotV3ArtifactError("v3 bound-file digest schema is incomplete")
    actual = _actual_hashes(root)
    if actual != recorded:
        raise NativePilotV3ArtifactError("a v3 bound-file digest does not match")

    previous_path = root / NATIVE_PILOT_LOCK_V2_PATH
    previous = _read_json(previous_path, "v2 draft lock")
    previous_raw = _sha256(previous_path)
    if (previous_raw != EXPECTED_V2_LOCK_RAW_SHA256
            or previous.get("approval_status") != "DRAFT_NOT_APPROVED"
            or _canonical_integrity(previous, "lock_canonical_sha256", "v2 predecessor lock")
                != EXPECTED_V2_LOCK_CANONICAL_SHA256):
        raise NativePilotV3ArtifactError("v2 predecessor lock changed or is not still draft")
    if lock.get("predecessor_v2_lock_raw_sha256") != previous_raw:
        raise NativePilotV3ArtifactError("v3 lock does not bind the expected v2 predecessor lock")

    original_path = root / ORIGINAL_LOCK_PATH
    original = _read_json(original_path, "original lock")
    original_raw = _sha256(original_path)
    if (original_raw != EXPECTED_ORIGINAL_LOCK_RAW_SHA256
            or original.get("approval_status") != "DRAFT_NOT_APPROVED"):
        raise NativePilotV3ArtifactError("original Phase 9.8 lock is not the unchanged draft")
    _canonical_integrity(original, "lock_sha256", "original lock")
    v1_path = root / NATIVE_V1_LOCK_PATH
    v1 = _read_json(v1_path, "native v1 lock")
    if v1.get("approval_status") != "DRAFT_NOT_APPROVED":
        raise NativePilotV3ArtifactError("native v1 predecessor lock is not still draft")
    _canonical_integrity(v1, "lock_sha256", "native v1 lock")

    inherited_keys = {
        "protocol", "configuration_table", "split_manifest", "model", "metadata",
        "training_report", "pilot_preregistration",
    }
    inherited = lock.get("inherited_frozen_raw_hashes")
    if (not isinstance(inherited, dict) or set(inherited) != inherited_keys
            or not isinstance(original.get("sha256"), dict)
            or {key: original["sha256"].get(key) for key in inherited_keys} != inherited):
        raise NativePilotV3ArtifactError("v3 inherited frozen hashes differ from the original lock")
    inherited_paths = {
        "protocol": "protocol", "configuration_table": "configuration_table",
        "split_manifest": "split_manifest", "model": "model", "metadata": "metadata",
        "training_report": "training_report", "pilot_preregistration": "original_preregistration",
    }
    if {key: actual[path_key] for key, path_key in inherited_paths.items()} != inherited:
        raise NativePilotV3ArtifactError("inherited frozen files differ from the original lock")
    if lock.get("inherited_original_lock_raw_sha256") != original_raw:
        raise NativePilotV3ArtifactError("v3 lock does not bind the original lock digest")

    try:
        design = load_frozen_pilot_design(root)
    except (OSError, ValueError) as exc:
        raise NativePilotV3ArtifactError("frozen pilot design validation failed") from exc
    group_ids = {spec[0] for spec in design.scenario_specs.values()}
    if len(group_ids) != 10 or len(design.scenario_specs) != 2560:
        raise NativePilotV3ArtifactError("pilot partition differs from the frozen 10-group/2560-scenario design")

    metadata = _read_json(root / METADATA_PATH, "model metadata")
    try:
        validate_metadata(metadata)
    except Phase98ArtifactError as exc:
        raise NativePilotV3ArtifactError("model metadata contract failed") from exc
    if (metadata.get("metadata_sha256") != _metadata_content_hash(metadata)
            or metadata.get("metadata_sha256") != lock.get("metadata_content_sha256")
            or metadata.get("artifact_sha256") != actual["model"]):
        raise NativePilotV3ArtifactError("model, metadata, or model-content binding differs")
    return metadata


def load_native_pilot_adapter_v3(
    *, root: Path | str = Path("."),
    lock_path: Path | str = NATIVE_PILOT_LOCK_V3_PATH,
    approved_lock_sha256: str | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> Phase98ProposedPolicyAdapter:
    project_root = Path(root)
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = project_root / lock_file
    if not lock_file.is_file() or not isinstance(approved_lock_sha256, str):
        raise NativePilotV3ArtifactError("v3 lock and independent raw digest are required")
    raw_hash = _sha256(lock_file)
    if raw_hash != approved_lock_sha256:
        raise NativePilotV3ArtifactError("raw v3 lock digest differs from independent expected value")
    lock = _read_json(lock_file, "v3 lock")
    metadata = _validate_v3_lock(lock, raw_hash, approved_lock_sha256, project_root)
    if model_loader is None:
        import joblib
        model_loader = joblib.load
    model = model_loader(project_root / MODEL_PATH)
    return Phase98ProposedPolicyAdapter(
        model, metadata, artifact_sha256=lock["bound_file_sha256"]["model"],
        _gate_token=_LOAD_GATE_TOKEN,
    )
