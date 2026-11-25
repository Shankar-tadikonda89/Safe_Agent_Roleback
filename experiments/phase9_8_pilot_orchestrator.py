"""Approval-gated Phase 9.8 pilot orchestration.

The v2 lock is validated and the model loaded before any scenario factory is
called. Potential outcomes are requested only after a persisted decision freeze
has been re-read and verified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.phase9_7_simulator import PotentialOutcomes
from experiments.phase9_8_adapter import Phase98ProposedPolicyAdapter
from experiments.phase9_8_decision_freeze import (
    FrozenPilotDesign, VerifiedDecisionFreeze, load_frozen_pilot_design,
    persist_decision_freeze, verify_decision_freeze,
)
from experiments.phase9_8_native_pilot_loader_v3 import (
    NATIVE_PILOT_LOCK_V3_PATH, NativePilotV3ArtifactError, load_native_pilot_adapter_v3,
)
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES, PilotScenario, Phase98PilotRunner,
    calculate_pilot_metrics, score_frozen_decisions, select_pilot_test_action,
)
from experiments.phase9_8_prefix_forecast import NativePrefixForecastContext


ScenarioFactory = Callable[[FrozenPilotDesign], tuple[Sequence[PilotScenario], Mapping[str, NativePrefixForecastContext]]]
OutcomeFactory = Callable[[Sequence[PilotScenario], VerifiedDecisionFreeze], Mapping[str, PotentialOutcomes]]
PILOT_RESULTS_V2_ROOT = Path("results/phase9_8_native_pilot_v2/runs")


@dataclass(frozen=True)
class ApprovedPilotResult:
    freeze_dir: Path
    freeze_manifest_sha256: str
    decision_count: int
    scored_rows: tuple[dict[str, Any], ...]
    metrics: Mapping[str, Any]
    pilot_test_selection: Mapping[str, Any]
    output_dir: Path


def _persist_pilot_artifacts(
    project_root: Path, freeze: VerifiedDecisionFreeze,
    scored: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any],
    pilot_selection: Mapping[str, Any],
) -> Path:
    """Create a versioned run directory; refuse to replace any prior result."""
    target = project_root / PILOT_RESULTS_V2_ROOT / f"run-{freeze.decision_records_sha256[:20]}"
    target.mkdir(parents=True, exist_ok=False)
    try:
        with (target / "comparative_scored_runs.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in scored:
                stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        with (target / "comparative_metrics.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(metrics), stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        with (target / "pilot_test_descriptive_selection.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(pilot_selection), stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    except OSError:
        # A partial create-only run directory remains unusable and is never reused.
        raise
    return target


def run_approved_native_pilot(
    *, approved_lock_sha256: str,
    scenario_factory: ScenarioFactory,
    evaluator_outcome_factory: OutcomeFactory,
    project_root: Path | str = Path("."),
    lock_path: Path | str = NATIVE_PILOT_LOCK_V3_PATH,
) -> ApprovedPilotResult:
    """Run only with an independently approved v2 lock digest.

    The adapter loader is intentionally invoked before design-dependent
    scenario generation. The outcome callback receives a verified freeze and
    is invoked only after decision records and freeze manifest pass re-read.
    """
    if (not isinstance(approved_lock_sha256, str) or len(approved_lock_sha256) != 64
            or any(char not in "0123456789abcdef" for char in approved_lock_sha256)):
        raise NativePilotV3ArtifactError("an independently approved raw lock SHA-256 is required")
    if not callable(scenario_factory) or not callable(evaluator_outcome_factory):
        raise TypeError("pilot orchestration requires scenario and evaluator factories")
    root = Path(project_root)

    adapter: Phase98ProposedPolicyAdapter = load_native_pilot_adapter_v3(
        root=root, lock_path=lock_path, approved_lock_sha256=approved_lock_sha256,
    )
    design = load_frozen_pilot_design(root)
    scenarios, contexts = scenario_factory(design)
    scenario_rows = list(scenarios)
    if not isinstance(contexts, Mapping) or set(contexts) != {
        row.policy_scenario.scenario_id for row in scenario_rows
    }:
        raise ValueError("scenario factory contexts must exactly match policy-visible scenarios")
    design.validate_scenarios(scenario_rows)
    for row in scenario_rows:
        context = contexts[row.policy_scenario.scenario_id]
        if (not isinstance(context, NativePrefixForecastContext)
                or context.scenario_id != row.policy_scenario.scenario_id
                or context.configuration_group_id != row.configuration_group_id
                or context.mechanism != row.mechanism
                or context.root_seed != row.root_seed):
            raise ValueError("native forecast context does not match the frozen scenario")

    decisions = Phase98PilotRunner().decide_matched(
        scenario_rows, proposed=adapter, native_prefix_contexts=contexts,
    )
    freeze_dir, retained_manifest_sha256 = persist_decision_freeze(
        decisions, scenario_rows, project_root=root,
    )
    verified = verify_decision_freeze(
        freeze_dir, scenario_rows, project_root=root,
        expected_manifest_sha256=retained_manifest_sha256,
    )

    # First evaluator callback: it receives only already-frozen decisions.
    outcomes = evaluator_outcome_factory(tuple(scenario_rows), verified)
    scored = score_frozen_decisions(
        verified.decisions, outcomes, expected_scenarios=scenario_rows,
    )
    metrics = calculate_pilot_metrics(scored)
    if set(metrics.get("policy_metrics", {})) != set(COMPARATIVE_POLICIES):
        raise ValueError("pilot metrics do not contain the exact amended four-policy set")

    # Outcome-selected pilot_test is descriptive only and cannot enter metrics.
    selection = select_pilot_test_action(outcomes).to_record()
    output_dir = _persist_pilot_artifacts(root, verified, scored, metrics, selection)
    return ApprovedPilotResult(
        freeze_dir=freeze_dir,
        freeze_manifest_sha256=verified.manifest_sha256,
        decision_count=len(verified.decisions),
        scored_rows=tuple(scored),
        metrics=metrics,
        pilot_test_selection=selection,
        output_dir=output_dir,
    )
