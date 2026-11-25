"""Create a new v3 draft lock binding the supersession document and lifecycle schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import METADATA_PATH, _canonical_bytes
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design
from experiments.phase9_8_native_pilot_loader_v2 import (
    EXPECTED_ORIGINAL_LOCK_RAW_SHA256, EXPECTED_POLICY_SET,
    NATIVE_PILOT_LOCK_V2_PATH, _read_json,
)
from experiments.phase9_8_native_pilot_loader_v3 import (
    EXPECTED_LOCK_SCHEMA_V3, NATIVE_PILOT_LOADER_V3_VERSION,
    NATIVE_PILOT_LOCK_V3_PATH, SUPERSEDING_PREREGISTRATION_V3_PATH,
    EXPECTED_V2_LOCK_RAW_SHA256, _actual_hashes,
)
from experiments.phase9_8_pilot_runner import COMPARATIVE_POLICIES
from experiments.phase9_5_observations import FEATURE_ORDER


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / NATIVE_PILOT_LOCK_V3_PATH
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    original_path = root / "results/phase9_8_pilot_lock.json"
    original = _read_json(original_path, "original lock")
    original_raw = hashlib.sha256(original_path.read_bytes()).hexdigest()
    if original_raw != EXPECTED_ORIGINAL_LOCK_RAW_SHA256 or original.get("approval_status") != "DRAFT_NOT_APPROVED":
        raise ValueError("original lock is not the unchanged draft")
    v2_path = root / NATIVE_PILOT_LOCK_V2_PATH
    v2 = _read_json(v2_path, "v2 predecessor lock")
    v2_raw = hashlib.sha256(v2_path.read_bytes()).hexdigest()
    if v2_raw != EXPECTED_V2_LOCK_RAW_SHA256 or v2.get("approval_status") != "DRAFT_NOT_APPROVED":
        raise ValueError("v2 predecessor lock is not the unchanged draft")
    if len(load_frozen_pilot_design(root).scenario_specs) != 2560:
        raise ValueError("pilot partition is not the frozen 2,560-scenario design")

    bound = _actual_hashes(root)
    inherited_keys = (
        "protocol", "configuration_table", "split_manifest", "model", "metadata",
        "training_report", "pilot_preregistration",
    )
    inherited = {key: original["sha256"][key] for key in inherited_keys}
    actual_inherited = {
        "protocol": bound["protocol"], "configuration_table": bound["configuration_table"],
        "split_manifest": bound["split_manifest"], "model": bound["model"],
        "metadata": bound["metadata"], "training_report": bound["training_report"],
        "pilot_preregistration": bound["original_preregistration"],
    }
    if actual_inherited != inherited:
        raise ValueError("inherited frozen artifact hashes differ")
    metadata = _read_json(root / METADATA_PATH, "model metadata")
    lock = {
        "schema_version": EXPECTED_LOCK_SCHEMA_V3,
        "approval_status": "DRAFT_NOT_APPROVED",
        "approval_note": "Draft only; no model load or pilot generation is authorized.",
        "loader_version": NATIVE_PILOT_LOADER_V3_VERSION,
        "superseding_document": str(SUPERSEDING_PREREGISTRATION_V3_PATH).replace("\\", "/"),
        "comparator_id": "phase9_8_native_prefix_forecast_v1",
        "comparative_policy_set": list(COMPARATIVE_POLICIES),
        "pilot_test_policy": {
            "in_comparative_set": False, "outcome_based_selection_after_freeze": True,
            "comparative_metric_eligible": False,
        },
        "frozen_feature_order": list(FEATURE_ORDER),
        "class_order": list(OUTCOME_CLASSES),
        "selected_C": 1.0,
        "metadata_content_sha256": metadata["metadata_sha256"],
        "forecast_count_per_action": 4,
        "forecast_count_total": 8,
        "expected_cost_rule": {
            "continue": "3 * (1 - p_pass_continue)",
            "intervene": "2 + 3 * (1 - p_pass_intervene)",
            "tie": "continue",
        },
        "execution_semantics": "one_decision_counterfactual_simulation",
        "pilot_scope": {
            "partition": "pilot", "configuration_group_count": 10,
            "scenario_count": 2560, "root_seed": 20269804,
        },
        "predecessor_v2_lock_raw_sha256": v2_raw,
        "inherited_original_lock_raw_sha256": original_raw,
        "inherited_frozen_raw_hashes": inherited,
        "bound_file_sha256": bound,
    }
    lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
    target.write_text(json.dumps(lock, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"lock={target.relative_to(root).as_posix()}")
    print(f"approval_status={lock['approval_status']}")
    print(f"raw_sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print(f"canonical_sha256={lock['lock_canonical_sha256']}")
    print(f"bound_files={len(bound)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
