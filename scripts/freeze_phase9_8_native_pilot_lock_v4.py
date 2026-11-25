"""Create a new DRAFT v4 lock for the wrapper-aware loader package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import EXPECTED_C, METADATA_PATH, _canonical_bytes
from experiments.phase9_8_native_pilot_loader_v2 import EXPECTED_POLICY_SET
from experiments.phase9_8_native_pilot_loader_v3 import (
    NATIVE_PILOT_LOCK_V3_PATH,
)
from experiments.phase9_8_native_pilot_loader_v4 import (
    NATIVE_PILOT_LOCK_V4_PATH, NATIVE_PILOT_LOADER_V4_VERSION,
    EXPECTED_LOCK_SCHEMA_V4, SUPERSEDING_DOCUMENT_V4_PATH,
    EXPECTED_V3_LOCK_RAW_SHA256, EXPECTED_V3_LOCK_CANONICAL_SHA256,
    _actual_hashes_v4, _validate_v3_lock,
)
from experiments.phase9_8_decision_freeze import load_frozen_pilot_design


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / NATIVE_PILOT_LOCK_V4_PATH
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if (previous.get("schema_version") != EXPECTED_LOCK_SCHEMA_V4
                or previous.get("approval_status") != "DRAFT_NOT_APPROVED"):
            raise FileExistsError(f"refusing to replace a non-draft v4 lock: {target}")

    v3_path = root / NATIVE_PILOT_LOCK_V3_PATH
    v3_raw = hashlib.sha256(v3_path.read_bytes()).hexdigest()
    if v3_raw != EXPECTED_V3_LOCK_RAW_SHA256:
        raise ValueError("approved v3 predecessor raw digest changed")
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    if v3.get("approval_status") != "APPROVED" or v3.get("lock_canonical_sha256") != EXPECTED_V3_LOCK_CANONICAL_SHA256:
        raise ValueError("v3 predecessor is not the approved frozen lock")
    _validate_v3_lock(v3, v3_raw, EXPECTED_V3_LOCK_RAW_SHA256, root)

    design = load_frozen_pilot_design(root)
    if (len(design.scenario_specs) != 2560
            or len({spec[0] for spec in design.scenario_specs.values()}) != 10
            or {spec[2] for spec in design.scenario_specs.values()} != {20269804}):
        raise ValueError("pilot partition differs from the approved 10-group design")
    metadata = json.loads((root / METADATA_PATH).read_text(encoding="utf-8"))
    bound = _actual_hashes_v4(root)
    lock = {
        "schema_version": EXPECTED_LOCK_SCHEMA_V4,
        "approval_status": "DRAFT_NOT_APPROVED",
        "approval_note": "Draft only; independently review and approve the v4 wrapper-loader package.",
        "loader_version": NATIVE_PILOT_LOADER_V4_VERSION,
        "superseding_document": str(SUPERSEDING_DOCUMENT_V4_PATH).replace("\\", "/"),
        "artifact_wrapper_format": "phase9.8-training-wrapper-v1",
        "comparator_id": "phase9_8_native_prefix_forecast_v1",
        "comparative_policy_set": list(EXPECTED_POLICY_SET),
        "pilot_test_policy": {
            "in_comparative_set": False,
            "outcome_based_selection_after_freeze": True,
            "comparative_metric_eligible": False,
        },
        "frozen_feature_order": list(FEATURE_ORDER),
        "class_order": list(OUTCOME_CLASSES),
        "selected_C": EXPECTED_C,
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
        "predecessor_v3_lock_raw_sha256": v3_raw,
        "predecessor_v3_lock_canonical_sha256": EXPECTED_V3_LOCK_CANONICAL_SHA256,
        "inherited_frozen_raw_hashes": dict(v3["inherited_frozen_raw_hashes"]),
        "bound_file_sha256": bound,
    }
    lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
    target.write_text(
        json.dumps(lock, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"lock={target.relative_to(root).as_posix()}")
    print(f"approval_status={lock['approval_status']}")
    print(f"raw_sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
    print(f"canonical_sha256={lock['lock_canonical_sha256']}")
    print(f"bound_files={len(bound)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
