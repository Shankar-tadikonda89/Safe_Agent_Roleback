"""Versioned, create-only decision freeze for the Phase 9.8 native pilot.

This is an integrity boundary, not a cryptographic signature or immutable
storage system. The caller must retain the returned raw manifest hash and
provide it when re-reading the freeze before the evaluator join.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_simulator import SIMULATOR_VERSION
from experiments.phase9_8_adapter import (
    CONFIGURATION_PATH,
    EXPECTED_CONFIGURATION_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SPLIT_SHA256,
    PROTOCOL_PATH,
    SPLIT_PATH,
    _canonical_bytes,
    _sha256,
)
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES,
    NATIVE_DECISION_SCHEMA_VERSION,
    PILOT_RUNNER_VERSION,
    PilotDecision,
    PilotScenario,
    _validate_complete_decision_set,
)


FREEZE_SCHEMA_VERSION = "phase9-8-pilot-decision-freeze-v2"
FROZEN_DECISION_SCHEMA_VERSION = "phase9-8-pilot-decision-record-v2"
FREEZE_MANIFEST_NAME = "freeze_manifest.json"
DECISION_RECORDS_NAME = "decisions.jsonl"
PILOT_FREEZE_ROOT = Path("results/phase9_8_pilot/decision_freezes")
PILOT_SCENARIOS_PER_GROUP = 256
EXPECTED_PILOT_SCENARIO_COUNT = 2560
PILOT_ROOT_SEED = 20269804
INITIAL_STATE_HASH = hashlib.sha256(b"phase9-7-empty-initial-state-v1").hexdigest()

# Raw bytes inherited from the frozen original Phase 9.8 lock. The separate
# canonical design hashes below are also checked against their file contents.
FROZEN_RAW_HASHES = {
    "protocol": "2858917ce9b4ac2bb25289d7f16cb5d71d467b95c1d59598a36ee034380e453e",
    "configuration_table": "a9e61597126e1bbe363e195f1654cf6903e674a8bf0c31edcae51a0404432188",
    "split_manifest": "79ea14414f2061ec1d185d03c52801d35fea60d7dd10ffe18d7d6f0f100b8473",
}
FROZEN_CANONICAL_HASHES = {
    "configuration_table": EXPECTED_CONFIGURATION_SHA256,
    "split_manifest": EXPECTED_SPLIT_SHA256,
}


class DecisionFreezeError(ValueError):
    """A persisted matched-decision set failed its freeze contract."""


def _canonical_json(value: Any) -> bytes:
    try:
        return _canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise DecisionFreezeError("decision freeze contains non-canonical JSON data") from exc


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scenario_id(root_seed: int, scenario_seed: int, mechanism: str) -> str:
    encoded = json.dumps(
        [SIMULATOR_VERSION, root_seed, scenario_seed, mechanism],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


@dataclass(frozen=True)
class FrozenPilotDesign:
    """Validated frozen pilot membership, containing no observations/outcomes."""

    scenario_specs: Mapping[str, tuple[str, str, int, int]]
    configuration_mechanisms: Mapping[str, str]
    configuration_parameters: Mapping[str, Mapping[str, float]]
    artifact_hashes: Mapping[str, str]

    def validate_scenarios(self, scenarios: Sequence[PilotScenario]) -> None:
        rows = list(scenarios)
        if not rows or any(not isinstance(row, PilotScenario) for row in rows):
            raise DecisionFreezeError("scenario set must contain PilotScenario records")
        ids = [row.policy_scenario.scenario_id for row in rows]
        if len(ids) != len(set(ids)):
            raise DecisionFreezeError("scenario set contains duplicate IDs")
        if set(ids) != set(self.scenario_specs):
            raise DecisionFreezeError("scenario set does not exactly cover the frozen pilot partition")
        for row in rows:
            scenario_id = row.policy_scenario.scenario_id
            group_id, mechanism, root_seed, scenario_seed = self.scenario_specs[scenario_id]
            if (row.configuration_group_id != group_id or row.mechanism != mechanism
                    or row.root_seed != root_seed or row.scenario_seed != scenario_seed
                    or scenario_id != _scenario_id(root_seed, scenario_seed, mechanism)
                    or row.policy_scenario.initial_state_hash != INITIAL_STATE_HASH):
                raise DecisionFreezeError("scenario provenance differs from the frozen pilot design")
            if self.configuration_mechanisms.get(group_id) != mechanism:
                raise DecisionFreezeError("scenario configuration group is unknown or has a different mechanism")
            if group_id not in self.configuration_parameters:
                raise DecisionFreezeError("scenario configuration parameters are missing")


def load_frozen_pilot_design(project_root: Path | str = Path(".")) -> FrozenPilotDesign:
    """Read and validate frozen design files without building scenarios."""
    root = Path(project_root)
    config_path = root / CONFIGURATION_PATH
    split_path = root / SPLIT_PATH
    protocol_path = root / PROTOCOL_PATH
    for name, path in (("protocol", protocol_path), ("configuration_table", config_path),
                       ("split_manifest", split_path)):
        if not path.is_file() or _sha256(path) != FROZEN_RAW_HASHES[name]:
            raise DecisionFreezeError(f"frozen {name} artifact is missing or changed")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionFreezeError("frozen design JSON is unreadable") from exc
    for name, document, field in (
        ("configuration_table", config, "configuration_table_sha256"),
        ("split_manifest", split, "split_manifest_sha256"),
    ):
        payload = dict(document)
        embedded = payload.pop(field, None)
        if (embedded != FROZEN_CANONICAL_HASHES[name]
                or _sha_bytes(_canonical_json(payload)) != embedded):
            raise DecisionFreezeError(f"frozen {name} canonical hash is invalid")
    if (config.get("configuration_table_sha256") != split.get("configuration_table_sha256")
            or config.get("configuration_table_sha256") != EXPECTED_CONFIGURATION_SHA256):
        raise DecisionFreezeError("configuration and split manifests do not match")

    configurations = config.get("configurations")
    groups = split.get("groups")
    if not isinstance(configurations, list) or not isinstance(groups, list):
        raise DecisionFreezeError("frozen design rows are malformed")
    configuration_mechanisms = {
        item.get("configuration_id"): item.get("mechanism")
        for item in configurations if isinstance(item, dict)
    }
    if len(configuration_mechanisms) != len(configurations):
        raise DecisionFreezeError("configuration IDs are duplicated or malformed")
    configuration_parameters: dict[str, dict[str, float]] = {}
    for item in configurations:
        group_id = item.get("configuration_id")
        params = item.get("parameters")
        if not isinstance(params, dict) or group_id not in configuration_mechanisms:
            raise DecisionFreezeError("configuration parameter row is malformed")
        configuration_parameters[group_id] = {str(k): float(v) for k, v in params.items()}

    pilot_groups = [item for item in groups if isinstance(item, dict) and item.get("partition") == "pilot"]
    if len(pilot_groups) != 10:
        raise DecisionFreezeError("frozen pilot partition must contain exactly 10 configuration groups")
    specs: dict[str, tuple[str, str, int, int]] = {}
    for group in pilot_groups:
        group_id = group.get("configuration_id")
        mechanism = group.get("mechanism")
        root_seed = group.get("partition_root_seed")
        start, end = group.get("scenario_seed_start_inclusive"), group.get("scenario_seed_end_inclusive")
        if (not isinstance(group_id, str) or configuration_mechanisms.get(group_id) != mechanism
                or type(root_seed) is not int or root_seed != PILOT_ROOT_SEED
                or type(start) is not int or type(end) is not int or end - start + 1 != PILOT_SCENARIOS_PER_GROUP
                or group.get("scenario_count") != PILOT_SCENARIOS_PER_GROUP):
            raise DecisionFreezeError("pilot group membership or seed range is invalid")
        for seed in range(start, end + 1):
            scenario_id = _scenario_id(root_seed, seed, mechanism)
            if scenario_id in specs:
                raise DecisionFreezeError("pilot scenario identity collision in frozen manifest")
            specs[scenario_id] = (group_id, mechanism, root_seed, seed)
    if len(specs) != EXPECTED_PILOT_SCENARIO_COUNT:
        raise DecisionFreezeError("frozen pilot scenario count is invalid")
    return FrozenPilotDesign(
        scenario_specs=specs,
        configuration_mechanisms=configuration_mechanisms,
        configuration_parameters=configuration_parameters,
        artifact_hashes=dict(FROZEN_RAW_HASHES),
    )


def _decision_record(decision: PilotDecision) -> dict[str, Any]:
    raw = decision.to_record()
    raw["schema_version"] = FROZEN_DECISION_SCHEMA_VERSION
    raw["source_schema_version"] = decision.to_record()["schema_version"]
    return raw


def _coverage_payload(scenarios: Sequence[PilotScenario]) -> list[dict[str, Any]]:
    return sorted(({
        "scenario_id": row.policy_scenario.scenario_id,
        "configuration_group_id": row.configuration_group_id,
        "mechanism": row.mechanism,
        "root_seed": row.root_seed,
        "scenario_seed": row.scenario_seed,
        "initial_state_hash": row.policy_scenario.initial_state_hash,
    } for row in scenarios), key=lambda item: item["scenario_id"])


def _record_bytes(decisions: Sequence[PilotDecision]) -> bytes:
    ordered = sorted(decisions, key=lambda row: (row.scenario_id, COMPARATIVE_POLICIES.index(row.policy)))
    try:
        return b"".join(_canonical_json(_decision_record(row)) + b"\n" for row in ordered)
    except (TypeError, ValueError) as exc:
        raise DecisionFreezeError("decision records contain invalid values") from exc


def _persist_decision_freeze(
    decisions: Sequence[PilotDecision], scenarios: Sequence[PilotScenario], *,
    design: FrozenPilotDesign, output_root: Path | str,
) -> tuple[Path, str]:
    """Persist complete decisions with create-only files; return folder and raw manifest hash."""
    scenario_rows = list(scenarios)
    decision_rows = list(decisions)
    design.validate_scenarios(scenario_rows)
    try:
        scenario_ids = _validate_complete_decision_set(decision_rows, scenario_rows)
    except (TypeError, ValueError) as exc:
        raise DecisionFreezeError("matched decisions are incomplete or inconsistent") from exc
    if len(scenario_ids) != len(design.scenario_specs):
        raise DecisionFreezeError("decision freeze requires complete frozen pilot coverage")
    if {row.policy for row in decision_rows} != set(COMPARATIVE_POLICIES):
        raise DecisionFreezeError("decision freeze requires exactly the four amended comparative policies")

    records = _record_bytes(decision_rows)
    records_hash = _sha_bytes(records)
    coverage_hash = _sha_bytes(_canonical_json(_coverage_payload(scenario_rows)))
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    freeze_dir = root / f"freeze-{records_hash[:20]}"
    try:
        freeze_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise DecisionFreezeError("decision freeze path already exists; refusing overwrite") from exc

    # Exclusive creation prevents accidental replacement. This is not an OS-
    # enforced immutable store; tampering is detected by the retained digest.
    record_path = freeze_dir / DECISION_RECORDS_NAME
    try:
        with record_path.open("xb") as stream:
            stream.write(records)
            stream.flush()
            os.fsync(stream.fileno())
        manifest: dict[str, Any] = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "decision_schema_version": FROZEN_DECISION_SCHEMA_VERSION,
            "source_decision_schema_versions": {
                "standard_policies": PILOT_RUNNER_VERSION,
                "native_prefix": NATIVE_DECISION_SCHEMA_VERSION,
            },
            "decision_records_file": DECISION_RECORDS_NAME,
            "decision_records_sha256": records_hash,
            "decision_record_count": len(decision_rows),
            "scenario_count": len(scenario_ids),
            "expected_decision_count": len(scenario_ids) * len(COMPARATIVE_POLICIES),
            "expected_policy_order": list(COMPARATIVE_POLICIES),
            "scenario_coverage_sha256": coverage_hash,
            "frozen_artifact_hashes": dict(design.artifact_hashes),
            "frozen_canonical_hashes": dict(FROZEN_CANONICAL_HASHES),
        }
        manifest["manifest_canonical_sha256"] = _sha_bytes(_canonical_json(manifest))
        manifest_bytes = _canonical_json(manifest) + b"\n"
        manifest_path = freeze_dir / FREEZE_MANIFEST_NAME
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise DecisionFreezeError("could not create the decision freeze; any partial directory is invalid") from exc
    return freeze_dir, _sha_bytes(manifest_bytes)


def persist_decision_freeze(
    decisions: Sequence[PilotDecision], scenarios: Sequence[PilotScenario], *,
    project_root: Path | str = Path("."),
) -> tuple[Path, str]:
    """Persist only against the on-disk frozen pilot design; no test contract injection."""
    design = load_frozen_pilot_design(project_root)
    if len(design.scenario_specs) != EXPECTED_PILOT_SCENARIO_COUNT:
        raise DecisionFreezeError("loaded frozen design has an unexpected pilot scenario count")
    if dict(design.artifact_hashes) != FROZEN_RAW_HASHES:
        raise DecisionFreezeError("frozen design artifact hash contract is invalid")
    return _persist_decision_freeze(
        decisions, scenarios, design=design,
        output_root=Path(project_root) / PILOT_FREEZE_ROOT,
    )


def _decision_from_record(record: Any) -> PilotDecision:
    if not isinstance(record, dict) or record.get("schema_version") != FROZEN_DECISION_SCHEMA_VERSION:
        raise DecisionFreezeError("frozen decision record schema is invalid")
    source_schema = record.get("source_schema_version")
    policy = record.get("policy")
    expected_source = NATIVE_DECISION_SCHEMA_VERSION if policy == COMPARATIVE_POLICIES[-1] else PILOT_RUNNER_VERSION
    if source_schema != expected_source:
        raise DecisionFreezeError("frozen decision source schema differs from policy")
    if policy == COMPARATIVE_POLICIES[-1] and record.get("comparator_id") != COMPARATIVE_POLICIES[-1]:
        raise DecisionFreezeError("native prefix decision lacks its canonical comparator identity")
    features = record.get("features")
    diagnostics = record.get("decision_diagnostics")
    if (record.get("feature_names") != list(FEATURE_ORDER) or not isinstance(features, dict)
            or set(features) != set(FEATURE_ORDER) or not isinstance(diagnostics, dict)):
        raise DecisionFreezeError("frozen decision feature or diagnostics schema is invalid")
    try:
        values = [features[name] for name in FEATURE_ORDER]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("feature values invalid")
        latency = record["local_decision_latency_seconds"]
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("latency invalid")
        return PilotDecision(
            scenario_id=record["scenario_id"],
            configuration_group_id=record["configuration_group_id"],
            mechanism=record["mechanism"], root_seed=record["root_seed"],
            scenario_seed=record["scenario_seed"], initial_state_hash=record["initial_state_hash"],
            policy=policy, selected_action=record["selected_action"],
            decision_sequence=record["decision_sequence"],
            features=tuple((name, float(features[name])) for name in FEATURE_ORDER),
            diagnostics=tuple(diagnostics.items()),
            local_decision_latency_seconds=float(latency),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionFreezeError("frozen decision record has invalid fields") from exc


@dataclass(frozen=True)
class VerifiedDecisionFreeze:
    """Decision records parsed from the exact bytes verified before the join."""

    freeze_dir: Path
    manifest_sha256: str
    decision_records_sha256: str
    decisions: tuple[PilotDecision, ...]


def _verify_decision_freeze(
    freeze_dir: Path | str, scenarios: Sequence[PilotScenario], *,
    design: FrozenPilotDesign, expected_manifest_sha256: str,
) -> VerifiedDecisionFreeze:
    """Re-read persisted decisions and manifest, validating against an external digest anchor."""
    directory = Path(freeze_dir)
    manifest_path = directory / FREEZE_MANIFEST_NAME
    records_path = directory / DECISION_RECORDS_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
        records_bytes = records_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecisionFreezeError("decision freeze files are missing or invalid") from exc
    raw_manifest_hash = _sha_bytes(manifest_bytes)
    if not expected_manifest_sha256 or raw_manifest_hash != expected_manifest_sha256:
        raise DecisionFreezeError("decision freeze manifest differs from its retained digest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise DecisionFreezeError("decision freeze manifest schema is invalid")
    payload = dict(manifest)
    canonical_hash = payload.pop("manifest_canonical_sha256", None)
    if canonical_hash != _sha_bytes(_canonical_json(payload)):
        raise DecisionFreezeError("decision freeze manifest canonical hash is invalid")
    if (_sha_bytes(records_bytes) != manifest.get("decision_records_sha256")
            or manifest.get("frozen_artifact_hashes") != dict(design.artifact_hashes)
            or manifest.get("frozen_canonical_hashes") != FROZEN_CANONICAL_HASHES
            or manifest.get("expected_policy_order") != list(COMPARATIVE_POLICIES)):
        raise DecisionFreezeError("decision records or frozen manifest bindings differ")

    try:
        lines = records_bytes.splitlines()
        if any(not line for line in lines):
            raise ValueError("empty JSONL line")
        records = [json.loads(line.decode("utf-8")) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DecisionFreezeError("decision JSONL is malformed") from exc
    decisions = tuple(_decision_from_record(record) for record in records)
    design.validate_scenarios(scenarios)
    try:
        scenario_ids = _validate_complete_decision_set(decisions, scenarios)
    except (TypeError, ValueError) as exc:
        raise DecisionFreezeError("persisted decisions fail matched coverage/provenance validation") from exc
    if ({row.policy for row in decisions} != set(COMPARATIVE_POLICIES)
            or len(decisions) != len(scenario_ids) * len(COMPARATIVE_POLICIES)
            or len(records) != manifest.get("decision_record_count")
            or len(scenario_ids) != manifest.get("scenario_count")
            or len(decisions) != manifest.get("expected_decision_count")
            or _sha_bytes(_canonical_json(_coverage_payload(scenarios))) != manifest.get("scenario_coverage_sha256")):
        raise DecisionFreezeError("persisted decision coverage differs from the manifest or pilot design")
    return VerifiedDecisionFreeze(directory, raw_manifest_hash,
                                  manifest["decision_records_sha256"], decisions)


def verify_decision_freeze(
    freeze_dir: Path | str, scenarios: Sequence[PilotScenario], *,
    expected_manifest_sha256: str, project_root: Path | str = Path("."),
) -> VerifiedDecisionFreeze:
    """Verify a freeze only against the actual frozen pilot contract on disk."""
    design = load_frozen_pilot_design(project_root)
    if len(design.scenario_specs) != EXPECTED_PILOT_SCENARIO_COUNT:
        raise DecisionFreezeError("loaded frozen design has an unexpected pilot scenario count")
    if dict(design.artifact_hashes) != FROZEN_RAW_HASHES:
        raise DecisionFreezeError("frozen design artifact hash contract is invalid")
    return _verify_decision_freeze(
        freeze_dir, scenarios, design=design,
        expected_manifest_sha256=expected_manifest_sha256,
    )
