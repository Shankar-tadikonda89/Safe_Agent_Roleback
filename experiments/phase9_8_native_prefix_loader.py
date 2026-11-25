"""Separate approval-gated loader for the Phase 9.8 native-prefix pilot lock.

The original Phase 9.8 adapter and loader are intentionally unchanged. This
loader validates the new comparator-specific lock independently and only
deserializes the fitted model after every lock, artifact, metadata, and split
check has passed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import (
    CONFIGURATION_PATH,
    EXPECTED_CONFIGURATION_SHA256,
    EXPECTED_C,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SPLIT_SHA256,
    METADATA_PATH,
    MODEL_PATH,
    PILOT_PREREG_PATH,
    PROTOCOL_PATH,
    SPLIT_PATH,
    TRAINING_REPORT_PATH,
    Phase98ArtifactError,
    Phase98ProposedPolicyAdapter,
    _LOAD_GATE_TOKEN,
    _canonical_bytes,
    _metadata_content_hash,
    _sha256,
    validate_metadata,
)


NATIVE_LOADER_VERSION = "phase9-8-native-prefix-loader-v1"
NATIVE_LOCK_PATH = Path("results/phase9_8_native_prefix_pilot_lock_v1.json")
NATIVE_ADDENDUM_PATH = Path("results/phase9_8_native_prefix_preregistration_addendum_v1.md")
NATIVE_FORECAST_PATH = Path("experiments/phase9_8_prefix_forecast.py")
PILOT_RUNNER_PATH = Path("experiments/phase9_8_pilot_runner.py")
NATIVE_FORECAST_TEST_PATH = Path("tests/test_phase9_8_prefix_forecast.py")
PILOT_RUNNER_TEST_PATH = Path("tests/test_phase9_8_pilot_runner.py")
NATIVE_LOADER_TEST_PATH = Path("tests/test_phase9_8_native_prefix_loader.py")
ORIGINAL_LOCK_PATH = Path("results/phase9_8_pilot_lock.json")
EXPECTED_ORIGINAL_LOCK_SHA256 = "ce5c5333f5d6258f6cc8a0dc871a9373c881ff16d8f54fc9b7b0e3c7d3444448"
EXPECTED_NATIVE_LOCK_SCHEMA = "phase9-8-native-prefix-independent-pilot-lock-v1"
COMPARATOR_VERSION = "phase9_8_native_prefix_forecast_v1"
EXPECTED_POLICY_SET = [
    "no_intervention", "naive_threshold", "proposed", "prefix_branching_approximation",
    COMPARATOR_VERSION,
]

_BOUND_PATHS = {
    "protocol": PROTOCOL_PATH,
    "configuration_table": CONFIGURATION_PATH,
    "split_manifest": SPLIT_PATH,
    "model": MODEL_PATH,
    "metadata": METADATA_PATH,
    "training_report": TRAINING_REPORT_PATH,
    "pilot_preregistration": PILOT_PREREG_PATH,
    "pilot_lock": ORIGINAL_LOCK_PATH,
    "native_prefix_addendum": NATIVE_ADDENDUM_PATH,
    "native_prefix_source": NATIVE_FORECAST_PATH,
    "pilot_runner_integration": PILOT_RUNNER_PATH,
    "native_prefix_tests": NATIVE_FORECAST_TEST_PATH,
    "pilot_runner_tests": PILOT_RUNNER_TEST_PATH,
    "native_prefix_loader": Path("experiments/phase9_8_native_prefix_loader.py"),
    "native_prefix_loader_tests": NATIVE_LOADER_TEST_PATH,
}
_INHERITED_KEYS = (
    "protocol", "configuration_table", "split_manifest", "model", "metadata",
    "training_report", "pilot_preregistration",
)


class NativePrefixArtifactError(Phase98ArtifactError):
    """New native-prefix lock or artifact failed validation."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativePrefixArtifactError(f"{label} is unreadable or malformed") from exc
    if not isinstance(value, dict):
        raise NativePrefixArtifactError(f"{label} must be a JSON object")
    return value


def _check_canonical_hash(document: dict[str, Any], field: str, label: str) -> None:
    payload = dict(document)
    recorded = payload.pop(field, None)
    if (not isinstance(recorded, str)
            or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != recorded):
        raise NativePrefixArtifactError(f"{label} canonical hash is invalid")


def _actual_bound_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, relative_path in _BOUND_PATHS.items():
        path = root / relative_path
        if not path.is_file():
            raise NativePrefixArtifactError(f"required bound artifact is missing: {name}")
        hashes[name] = _sha256(path)
    return hashes


def _validate_native_lock(lock: dict[str, Any], lock_raw_sha256: str,
                          approved_lock_sha256: str, root: Path) -> dict[str, Any]:
    if lock_raw_sha256 != approved_lock_sha256:
        raise NativePrefixArtifactError("native-prefix lock raw hash is not independently approved")
    if (lock.get("schema_version") != EXPECTED_NATIVE_LOCK_SCHEMA
            or lock.get("approval_status") != "APPROVED"
            or lock.get("comparator_version") != COMPARATOR_VERSION):
        raise NativePrefixArtifactError("native-prefix lock is not approved for this comparator")
    _check_canonical_hash(lock, "lock_sha256", "native-prefix lock")
    expected_fields = {
        "loader_version": NATIVE_LOADER_VERSION,
        "forecast_count_per_decision": 8,
        "action_forecasts": 4,
        "frozen_feature_order": list(FEATURE_ORDER),
        "matched_decision_policy_set": EXPECTED_POLICY_SET,
        "pilot_scope": {"partition": "pilot", "root_seed": 20269804, "scenario_count": 2560},
    }
    if any(lock.get(key) != value for key, value in expected_fields.items()):
        raise NativePrefixArtifactError("native-prefix lock policy or pilot scope differs from the reviewed contract")

    hashes = lock.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_BOUND_PATHS):
        raise NativePrefixArtifactError("native-prefix lock bound-hash schema is invalid")
    actual_hashes = _actual_bound_hashes(root)
    if hashes != actual_hashes:
        raise NativePrefixArtifactError("a bound native-prefix artifact hash differs from the lock")

    original_lock_path = root / ORIGINAL_LOCK_PATH
    original_hash = actual_hashes["pilot_lock"]
    if (original_hash != EXPECTED_ORIGINAL_LOCK_SHA256
            or lock.get("inherited_draft_lock_sha256") != original_hash):
        raise NativePrefixArtifactError("inherited original draft lock does not match its frozen hash")
    original = _read_json(original_lock_path, "original pilot lock")
    if (original.get("schema_version") != "phase9-8-independent-pilot-lock-v1"
            or original.get("approval_status") != "DRAFT_NOT_APPROVED"):
        raise NativePrefixArtifactError("original pilot lock changed or is not still draft")
    _check_canonical_hash(original, "lock_sha256", "original pilot lock")
    inherited = lock.get("inherited_frozen_raw_hashes")
    original_hashes = original.get("sha256")
    if (not isinstance(inherited, dict) or set(inherited) != set(_INHERITED_KEYS)
            or not isinstance(original_hashes, dict)
            or {key: original_hashes.get(key) for key in _INHERITED_KEYS} != inherited
            or {key: actual_hashes[key] for key in _INHERITED_KEYS} != inherited):
        raise NativePrefixArtifactError("inherited frozen artifact hashes do not match the original lock")
    if lock.get("inherited_canonical_hashes") != original.get("canonical_design_hashes"):
        raise NativePrefixArtifactError("inherited canonical design hashes differ from original lock")

    config = _read_json(root / CONFIGURATION_PATH, "configuration table")
    split = _read_json(root / SPLIT_PATH, "split manifest")
    if (config.get("configuration_table_sha256") != EXPECTED_CONFIGURATION_SHA256
            or split.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or split.get("configuration_table_sha256") != EXPECTED_CONFIGURATION_SHA256
            or EXPECTED_PROTOCOL_SHA256 != inherited.get("protocol")):
        raise NativePrefixArtifactError("frozen protocol or design hashes are inconsistent")
    groups = [group for group in split.get("groups", []) if group.get("partition") == "pilot"]
    group_ids = [group.get("configuration_id") for group in groups]
    scenario_count = sum(group.get("scenario_count", 0) for group in groups)
    original_pilot = original.get("pilot_partition")
    if (len(groups) != 10 or scenario_count != 2560
            or not isinstance(original_pilot, dict)
            or original_pilot.get("root_seed") != 20269804
            or original_pilot.get("scenario_count") != scenario_count
            or original_pilot.get("configuration_group_ids") != group_ids):
        raise NativePrefixArtifactError("pilot partition does not match the frozen split and original lock")

    metadata = _read_json(root / METADATA_PATH, "model metadata")
    try:
        validate_metadata(metadata)
    except Phase98ArtifactError as exc:
        raise NativePrefixArtifactError("model metadata does not match the frozen contract") from exc
    metadata_canonical = _metadata_content_hash(metadata)
    if (metadata.get("metadata_sha256") != metadata_canonical
            or metadata.get("metadata_sha256") != lock.get("inherited_canonical_hashes", {}).get("metadata_content_sha256")
            or metadata.get("artifact_sha256") != actual_hashes["model"]
            or lock.get("selected_C", EXPECTED_C) != EXPECTED_C):
        raise NativePrefixArtifactError("model metadata, model digest, or selected hyperparameter is inconsistent")
    return metadata


def load_native_prefix_adapter(
    *, root: Path | str = Path("."), lock_path: Path | str = NATIVE_LOCK_PATH,
    approved_lock_sha256: str | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> Phase98ProposedPolicyAdapter:
    """Load only after the separately approved native-prefix lock validates.

    Production callers must provide the independently approved raw lock hash.
    Tests may inject a fake loader; the project model is never loaded by tests.
    """
    project_root = Path(root)
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = project_root / lock_file
    if not lock_file.is_file():
        raise NativePrefixArtifactError("native-prefix pilot lock is missing")
    lock_raw_hash = _sha256(lock_file)
    if not approved_lock_sha256 or lock_raw_hash != approved_lock_sha256:
        raise NativePrefixArtifactError("native-prefix lock raw hash is not independently approved")
    lock = _read_json(lock_file, "native-prefix lock")
    metadata = _validate_native_lock(lock, lock_raw_hash, approved_lock_sha256, project_root)
    model_path = project_root / MODEL_PATH
    if model_loader is None:
        import joblib
        model_loader = joblib.load
    model = model_loader(model_path)
    return Phase98ProposedPolicyAdapter(
        model, metadata, artifact_sha256=lock["sha256"]["model"],
        _gate_token=_LOAD_GATE_TOKEN,
    )
