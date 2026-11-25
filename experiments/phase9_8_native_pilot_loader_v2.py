"""Independent fail-closed loader for the Phase 9.8 adoption-amendment lock.

This v2 gate is separate from, and does not weaken, either earlier loader.
Model deserialization is the final operation after lock approval, hashes,
inherited artifacts, metadata, and pilot partition all validate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import (
    CONFIGURATION_PATH, EXPECTED_C, EXPECTED_CONFIGURATION_SHA256,
    EXPECTED_PROTOCOL_SHA256, EXPECTED_SPLIT_SHA256, METADATA_PATH, MODEL_PATH,
    PILOT_PREREG_PATH, PROTOCOL_PATH, SPLIT_PATH, TRAINING_REPORT_PATH,
    Phase98ArtifactError, Phase98ProposedPolicyAdapter, _LOAD_GATE_TOKEN,
    _canonical_bytes, _metadata_content_hash, _sha256, validate_metadata,
)
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design
from experiments.phase9_8_pilot_runner import COMPARATIVE_POLICIES


NATIVE_PILOT_LOADER_V2_VERSION = "phase9-8-native-pilot-loader-v2"
NATIVE_PILOT_LOCK_V2_PATH = Path("results/phase9_8_native_pilot_lock_v2.json")
NATIVE_PILOT_AMENDMENT_V2_PATH = Path("results/phase9_8_native_prefix_adoption_amendment_v2.md")
NATIVE_PILOT_ENTRYPOINT_PATH = Path("experiments/phase9_8_native_pilot.py")
NATIVE_PILOT_FACTORY_PATH = Path("experiments/phase9_8_pilot_factories.py")
ORIGINAL_LOCK_PATH = Path("results/phase9_8_pilot_lock.json")
NATIVE_V1_LOCK_PATH = Path("results/phase9_8_native_prefix_pilot_lock_v1.json")
NATIVE_V1_ADDENDUM_PATH = Path("results/phase9_8_native_prefix_preregistration_addendum_v1.md")
NATIVE_V1_LOADER_PATH = Path("experiments/phase9_8_native_prefix_loader.py")
EXPECTED_ORIGINAL_LOCK_RAW_SHA256 = "ce5c5333f5d6258f6cc8a0dc871a9373c881ff16d8f54fc9b7b0e3c7d3444448"
EXPECTED_LOCK_SCHEMA = "phase9-8-native-pilot-independent-lock-v2"
COMPARATOR_ID = "phase9_8_native_prefix_forecast_v1"
EXPECTED_POLICY_SET = ["no_intervention", "naive_threshold", "proposed", COMPARATOR_ID]

_BOUND_PATHS = {
    "adoption_amendment": NATIVE_PILOT_AMENDMENT_V2_PATH,
    "entrypoint": NATIVE_PILOT_ENTRYPOINT_PATH,
    "orchestration": Path("experiments/phase9_8_pilot_orchestrator.py"),
    "decision_freeze": Path("experiments/phase9_8_decision_freeze.py"),
    "scenario_factories": NATIVE_PILOT_FACTORY_PATH,
    "adapter": Path("experiments/phase9_8_adapter.py"),
    "pilot_runner": Path("experiments/phase9_8_pilot_runner.py"),
    "forecast_provider": Path("experiments/phase9_8_prefix_forecast.py"),
    "native_loader_v2": Path("experiments/phase9_8_native_pilot_loader_v2.py"),
    "lock_builder": Path("scripts/freeze_phase9_8_native_pilot_lock_v2.py"),
    "native_loader_v1": NATIVE_V1_LOADER_PATH,
    "native_v1_addendum": NATIVE_V1_ADDENDUM_PATH,
    "native_v1_lock": NATIVE_V1_LOCK_PATH,
    "protocol": PROTOCOL_PATH,
    "configuration_table": CONFIGURATION_PATH,
    "split_manifest": SPLIT_PATH,
    "model": MODEL_PATH,
    "metadata": METADATA_PATH,
    "training_report": TRAINING_REPORT_PATH,
    "original_preregistration": PILOT_PREREG_PATH,
    "original_lock": ORIGINAL_LOCK_PATH,
    "phase9_7_simulator": Path("experiments/phase9_7_simulator.py"),
    "phase9_8_training": Path("experiments/phase9_8_training.py"),
    "pilot_runner_tests": Path("tests/test_phase9_8_pilot_runner.py"),
    "decision_freeze_tests": Path("tests/test_phase9_8_decision_freeze.py"),
    "factory_tests": Path("tests/test_phase9_8_pilot_factories.py"),
    "orchestrator_tests": Path("tests/test_phase9_8_pilot_orchestrator.py"),
    "loader_tests": Path("tests/test_phase9_8_native_pilot_loader_v2.py"),
    "forecast_tests": Path("tests/test_phase9_8_prefix_forecast.py"),
    "legacy_native_loader_tests": Path("tests/test_phase9_8_native_prefix_loader.py"),
    "adapter_regression_tests": Path("tests/test_phase9_8_adapter.py"),
    "training_protocol_tests": Path("tests/test_phase9_8_training.py"),
    "phase9_7_regression_tests": Path("tests/test_phase9_7_integration.py"),
}


class NativePilotArtifactError(Phase98ArtifactError):
    """The independently approved Phase 9.8 v2 pilot lock failed validation."""


def _canonical_hash(document: dict[str, Any], field: str, label: str) -> None:
    payload = dict(document)
    recorded = payload.pop(field, None)
    if (not isinstance(recorded, str)
            or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != recorded):
        raise NativePilotArtifactError(f"{label} canonical hash is invalid")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativePilotArtifactError(f"{label} is unreadable") from exc
    if not isinstance(result, dict):
        raise NativePilotArtifactError(f"{label} has invalid schema")
    return result


def _actual_hashes(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, relative in _BOUND_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise NativePilotArtifactError(f"required bound file is missing: {name}")
        values[name] = _sha256(path)
    return values


def _validate_lock(lock: dict[str, Any], raw_hash: str,
                   approved_hash: str, root: Path) -> dict[str, Any]:
    if raw_hash != approved_hash:
        raise NativePilotArtifactError("raw lock digest differs from independently supplied digest")
    if (lock.get("schema_version") != EXPECTED_LOCK_SCHEMA
            or lock.get("approval_status") != "APPROVED"
            or lock.get("comparator_id") != COMPARATOR_ID
            or lock.get("loader_version") != NATIVE_PILOT_LOADER_V2_VERSION):
        raise NativePilotArtifactError("v2 pilot lock is draft or has an unsupported contract")
    _canonical_hash(lock, "lock_canonical_sha256", "v2 pilot lock")
    if lock.get("comparative_policy_set") != EXPECTED_POLICY_SET:
        raise NativePilotArtifactError("v2 comparative policy set differs from the amendment")
    if lock.get("pilot_test_policy") != {
        "in_comparative_set": False, "outcome_based_selection_after_freeze": True,
        "comparative_metric_eligible": False,
    }:
        raise NativePilotArtifactError("pilot_test inclusion semantics differ from the amendment")
    if lock.get("frozen_feature_order") != list(FEATURE_ORDER):
        raise NativePilotArtifactError("v2 feature order differs from the frozen model schema")
    if lock.get("pilot_scope") != {
        "partition": "pilot", "configuration_group_count": 10,
        "scenario_count": 2560, "root_seed": 20269804,
    }:
        raise NativePilotArtifactError("v2 pilot scope differs from the frozen split")
    if (lock.get("forecast_count_per_action") != 4
            or lock.get("forecast_count_total") != 8
            or lock.get("expected_cost_rule") != {
                "continue": "3 * (1 - p_pass_continue)",
                "intervene": "2 + 3 * (1 - p_pass_intervene)",
                "tie": "continue",
            }):
        raise NativePilotArtifactError("forecast count or expected-cost rule differs")

    recorded = lock.get("bound_file_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(_BOUND_PATHS):
        raise NativePilotArtifactError("v2 bound-file hash map is incomplete")
    actual = _actual_hashes(root)
    if actual != recorded:
        raise NativePilotArtifactError("a bound-file digest differs from the v2 lock")

    original = _read_json(root / ORIGINAL_LOCK_PATH, "original lock")
    original_raw_hash = _sha256(root / ORIGINAL_LOCK_PATH)
    if (original_raw_hash != EXPECTED_ORIGINAL_LOCK_RAW_SHA256
            or lock.get("inherited_original_lock_raw_sha256") != original_raw_hash
            or original.get("approval_status") != "DRAFT_NOT_APPROVED"
            or original.get("schema_version") != "phase9-8-independent-pilot-lock-v1"):
        raise NativePilotArtifactError("original lock is not the unchanged draft")
    _canonical_hash(original, "lock_sha256", "original lock")
    inherited = lock.get("inherited_frozen_raw_hashes")
    orig_hashes = original.get("sha256")
    inherited_keys = {"protocol", "configuration_table", "split_manifest", "model",
                      "metadata", "training_report", "pilot_preregistration"}
    if (not isinstance(inherited, dict) or set(inherited) != inherited_keys
            or not isinstance(orig_hashes, dict)
            or {key: orig_hashes.get(key) for key in inherited_keys} != inherited):
        raise NativePilotArtifactError("inherited frozen hashes differ from the original lock")
    path_keys = {
        "protocol": "protocol", "configuration_table": "configuration_table",
        "split_manifest": "split_manifest", "model": "model", "metadata": "metadata",
        "training_report": "training_report", "pilot_preregistration": "original_preregistration",
    }
    if {key: actual[value] for key, value in path_keys.items()} != inherited:
        raise NativePilotArtifactError("inherited artifact files do not match frozen raw hashes")
    if lock.get("inherited_canonical_hashes") != original.get("canonical_design_hashes"):
        raise NativePilotArtifactError("inherited canonical design hashes differ")

    native_v1 = _read_json(root / NATIVE_V1_LOCK_PATH, "native v1 lock")
    if (native_v1.get("approval_status") != "DRAFT_NOT_APPROVED"
            or native_v1.get("schema_version") != "phase9-8-native-prefix-independent-pilot-lock-v1"):
        raise NativePilotArtifactError("prior native v1 lock is not still draft")
    _canonical_hash(native_v1, "lock_sha256", "native v1 lock")

    try:
        design = load_frozen_pilot_design(root)
    except (OSError, ValueError) as exc:
        raise NativePilotArtifactError("frozen protocol/configuration/split validation failed") from exc
    original_partition = original.get("pilot_partition")
    groups = {spec[0] for spec in design.scenario_specs.values()}
    if (len(groups) != 10 or len(design.scenario_specs) != 2560
            or not isinstance(original_partition, dict)
            or original_partition.get("configuration_group_ids") != [
                row["configuration_id"] for row in _read_json(root / SPLIT_PATH, "split manifest")["groups"]
                if row.get("partition") == "pilot"
            ]):
        raise NativePilotArtifactError("pilot partition membership differs from the frozen split")

    metadata = _read_json(root / METADATA_PATH, "model metadata")
    try:
        validate_metadata(metadata)
    except Phase98ArtifactError as exc:
        raise NativePilotArtifactError("model metadata contract failed") from exc
    if (metadata.get("metadata_sha256") != _metadata_content_hash(metadata)
            or metadata.get("metadata_sha256") != lock.get("metadata_content_sha256")
            or metadata.get("artifact_sha256") != actual["model"]
            or lock.get("selected_C") != EXPECTED_C
            or lock.get("frozen_feature_order") != list(FEATURE_ORDER)
            or lock.get("class_order") != list(OUTCOME_CLASSES)):
        raise NativePilotArtifactError("metadata, artifact, hyperparameter, feature, or class order differs")
    return metadata


def load_native_pilot_adapter_v2(
    *, root: Path | str = Path("."),
    lock_path: Path | str = NATIVE_PILOT_LOCK_V2_PATH,
    approved_lock_sha256: str | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> Phase98ProposedPolicyAdapter:
    """Validate all approval bindings before the single deserialization step."""
    project_root = Path(root)
    lock_file = Path(lock_path)
    if not lock_file.is_absolute():
        lock_file = project_root / lock_file
    if not lock_file.is_file() or not isinstance(approved_lock_sha256, str):
        raise NativePilotArtifactError("approved v2 lock and independently supplied digest are required")
    raw_hash = _sha256(lock_file)
    if raw_hash != approved_lock_sha256:
        raise NativePilotArtifactError("raw lock digest differs from independently supplied digest")
    lock = _read_json(lock_file, "v2 pilot lock")
    metadata = _validate_lock(lock, raw_hash, approved_lock_sha256, project_root)
    if model_loader is None:
        import joblib
        model_loader = joblib.load
    model = model_loader(project_root / MODEL_PATH)
    return Phase98ProposedPolicyAdapter(
        model, metadata, artifact_sha256=lock["bound_file_sha256"]["model"],
        _gate_token=_LOAD_GATE_TOKEN,
    )
