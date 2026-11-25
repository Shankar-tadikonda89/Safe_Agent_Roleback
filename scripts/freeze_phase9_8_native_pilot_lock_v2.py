"""Create the versioned v2 draft lock without approval or model loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.phase9_8_adapter import METADATA_PATH, _canonical_bytes
from experiments.phase9_8_native_pilot_loader_v2 import (
    COMPARATOR_ID, EXPECTED_ORIGINAL_LOCK_RAW_SHA256, EXPECTED_POLICY_SET,
    EXPECTED_LOCK_SCHEMA, NATIVE_PILOT_AMENDMENT_V2_PATH,
    NATIVE_PILOT_LOCK_V2_PATH, NATIVE_PILOT_LOADER_V2_VERSION,
    _actual_hashes, _read_json,
)
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design
from experiments.phase9_8_pilot_runner import COMPARATIVE_POLICIES
from experiments.phase9_7_runner import OUTCOME_CLASSES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / NATIVE_PILOT_LOCK_V2_PATH
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    original = _read_json(root / "results/phase9_8_pilot_lock.json", "original lock")
    original_raw = hashlib.sha256((root / "results/phase9_8_pilot_lock.json").read_bytes()).hexdigest()
    if original_raw != EXPECTED_ORIGINAL_LOCK_RAW_SHA256 or original.get("approval_status") != "DRAFT_NOT_APPROVED":
        raise ValueError("the original frozen lock is not the expected unchanged draft")
    v1 = _read_json(root / "results/phase9_8_native_prefix_pilot_lock_v1.json", "native v1 lock")
    if v1.get("approval_status") != "DRAFT_NOT_APPROVED":
        raise ValueError("the prior native-prefix lock must remain draft")
    design = load_frozen_pilot_design(root)
    if len(design.scenario_specs) != 2560 or len({spec[0] for spec in design.scenario_specs.values()}) != 10:
        raise ValueError("frozen pilot partition does not have the approved 10/2560 membership")

    bound = _actual_hashes(root)
    inherited_keys = (
        "protocol", "configuration_table", "split_manifest", "model", "metadata",
        "training_report", "pilot_preregistration",
    )
    inherited = {key: original["sha256"][key] for key in inherited_keys}
    actual_by_original_name = {
        "protocol": bound["protocol"], "configuration_table": bound["configuration_table"],
        "split_manifest": bound["split_manifest"], "model": bound["model"],
        "metadata": bound["metadata"], "training_report": bound["training_report"],
        "pilot_preregistration": bound["original_preregistration"],
    }
    if actual_by_original_name != inherited:
        raise ValueError("inherited frozen artifact hashes differ; draft lock not created")
    metadata = _read_json(root / METADATA_PATH, "model metadata")
    lock = {
        "schema_version": EXPECTED_LOCK_SCHEMA,
        "approval_status": "DRAFT_NOT_APPROVED",
        "approval_note": "Draft only; does not authorize model deserialization, scenario generation, or pilot execution.",
        "loader_version": NATIVE_PILOT_LOADER_V2_VERSION,
        "comparator_id": COMPARATOR_ID,
        "comparative_policy_set": list(COMPARATIVE_POLICIES),
        "pilot_test_policy": {
            "in_comparative_set": False, "outcome_based_selection_after_freeze": True,
            "comparative_metric_eligible": False,
        },
        "frozen_feature_order": ["failure_prob", "reversibility_proxy", "trajectory_position", "disruption_history"],
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
        "pilot_scope": {
            "partition": "pilot", "configuration_group_count": 10,
            "scenario_count": 2560, "root_seed": 20269804,
        },
        "inherited_original_lock_raw_sha256": original_raw,
        "inherited_frozen_raw_hashes": inherited,
        "inherited_canonical_hashes": original["canonical_design_hashes"],
        "amendment_path": str(NATIVE_PILOT_AMENDMENT_V2_PATH).replace("\\", "/"),
        "bound_file_sha256": bound,
    }
    lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lock, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    raw = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"lock={target.relative_to(root).as_posix()}")
    print(f"approval_status={lock['approval_status']}")
    print(f"raw_sha256={raw}")
    print(f"canonical_sha256={lock['lock_canonical_sha256']}")
    print(f"bound_files={len(bound)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
