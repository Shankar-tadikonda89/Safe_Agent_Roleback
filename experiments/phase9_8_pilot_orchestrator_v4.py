"""v4 pilot orchestration using the wrapper-aware approval-gated loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.phase9_7_simulator import PotentialOutcomes
from experiments.phase9_8_adapter import Phase98ProposedPolicyAdapter
from experiments.phase9_8_decision_freeze import (
    VerifiedDecisionFreeze, load_frozen_pilot_design, persist_decision_freeze,
    verify_decision_freeze,
)
from experiments.phase9_8_native_pilot_loader_v4 import (
    NATIVE_PILOT_LOCK_V4_PATH, NativePilotV4ArtifactError,
    load_native_pilot_adapter_v4,
)
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES, PilotScenario, Phase98PilotRunner,
    calculate_pilot_metrics, score_frozen_decisions, select_pilot_test_action,
)
from experiments.phase9_8_prefix_forecast import NativePrefixForecastContext


V4_RESULTS_ROOT = Path("results/phase9_8_native_pilot_v4/runs")
ScenarioFactory = Callable[[Any], tuple[Sequence[PilotScenario], Mapping[str, NativePrefixForecastContext]]]
OutcomeFactory = Callable[[Sequence[PilotScenario], VerifiedDecisionFreeze], Mapping[str, PotentialOutcomes]]


@dataclass(frozen=True)
class ApprovedPilotV4Result:
    freeze_dir: Path
    freeze_manifest_sha256: str
    decision_count: int
    scored_rows: tuple[dict[str, Any], ...]
    metrics: Mapping[str, Any]
    pilot_test_selection: Mapping[str, Any]
    output_dir: Path


def _persist_results(
    project_root: Path, freeze: VerifiedDecisionFreeze,
    scored: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any],
    pilot_selection: Mapping[str, Any],
) -> Path:
    target = project_root / V4_RESULTS_ROOT / f"run-{freeze.decision_records_sha256[:20]}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "comparative_scored_runs.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in scored:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    with (target / "comparative_metrics.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(metrics), stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    with (target / "pilot_test_descriptive_selection.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(pilot_selection), stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    return target


def run_approved_native_pilot_v4(
    *, approved_lock_sha256: str, scenario_factory: ScenarioFactory,
    evaluator_outcome_factory: OutcomeFactory, project_root: Path | str = Path("."),
    lock_path: Path | str = NATIVE_PILOT_LOCK_V4_PATH,
) -> ApprovedPilotV4Result:
    """Load only through v4; generate outcomes after persisted freeze verification."""
    if (not isinstance(approved_lock_sha256, str) or len(approved_lock_sha256) != 64
            or any(char not in "0123456789abcdef" for char in approved_lock_sha256)):
        raise NativePilotV4ArtifactError("an independent approved v4 lock SHA-256 is required")
    if not callable(scenario_factory) or not callable(evaluator_outcome_factory):
        raise TypeError("pilot orchestration requires scenario and evaluator factories")
    root = Path(project_root)
    adapter: Phase98ProposedPolicyAdapter = load_native_pilot_adapter_v4(
        root=root, lock_path=lock_path, approved_lock_sha256=approved_lock_sha256,
    )

    design = load_frozen_pilot_design(root)
    scenarios, contexts = scenario_factory(design)
    rows = list(scenarios)
    if not isinstance(contexts, Mapping) or set(contexts) != {
        row.policy_scenario.scenario_id for row in rows
    }:
        raise ValueError("scenario factory contexts must exactly match policy scenarios")
    design.validate_scenarios(rows)
    for row in rows:
        context = contexts[row.policy_scenario.scenario_id]
        if (not isinstance(context, NativePrefixForecastContext)
                or context.scenario_id != row.policy_scenario.scenario_id
                or context.configuration_group_id != row.configuration_group_id
                or context.mechanism != row.mechanism
                or context.root_seed != row.root_seed):
            raise ValueError("native forecast context does not match frozen scenario")

    decisions = Phase98PilotRunner().decide_matched(
        rows, proposed=adapter, native_prefix_contexts=contexts,
    )
    freeze_dir, retained_digest = persist_decision_freeze(decisions, rows, project_root=root)
    verified = verify_decision_freeze(
        freeze_dir, rows, project_root=root, expected_manifest_sha256=retained_digest,
    )
    outcomes = evaluator_outcome_factory(tuple(rows), verified)
    scored = score_frozen_decisions(verified.decisions, outcomes, expected_scenarios=rows)
    metrics = calculate_pilot_metrics(scored)
    if set(metrics.get("policy_metrics", {})) != set(COMPARATIVE_POLICIES):
        raise ValueError("pilot metrics differ from the frozen comparative policy set")
    selection = select_pilot_test_action(outcomes).to_record()
    output_dir = _persist_results(root, verified, scored, metrics, selection)
    return ApprovedPilotV4Result(
        freeze_dir=freeze_dir, freeze_manifest_sha256=verified.manifest_sha256,
        decision_count=len(verified.decisions), scored_rows=tuple(scored),
        metrics=metrics, pilot_test_selection=selection, output_dir=output_dir,
    )
