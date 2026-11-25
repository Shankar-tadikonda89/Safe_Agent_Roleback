"""Offline dashboard helpers that reuse the project execution and metrics code."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from experiments.metrics import POLICIES, compute_metrics, load_run_summaries
from policy.proposed import DEFAULT_ARTIFACT
from run_demo import Planner, TASK_BY_ID, TASKS, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = PROJECT_ROOT / "logs" / "dashboard_offline_runs.jsonl"
DEFAULT_METRICS_LOG = PROJECT_ROOT / "results" / "heldout_evaluation" / "evaluation_runs.jsonl"
PHASE98_RUN_ID = "run-3d4f75d0ae811b9a7342"
PHASE98_RESULTS_DIR = Path("results/phase9_8_native_pilot_v4/runs") / PHASE98_RUN_ID
PHASE98_AUDIT_PATH = Path("reports/finalization/post_pilot_audit.md")
PHASE98_POLICIES = (
    "no_intervention", "naive_threshold", "proposed",
    "phase9_8_native_prefix_forecast_v1",
)
PHASE98_POLICY_LABELS = {
    "no_intervention": "No intervention",
    "naive_threshold": "Naive threshold",
    "proposed": "Proposed (Phase 9.8)",
    "phase9_8_native_prefix_forecast_v1": "Prefix comparator",
}
RESEARCH_LIMITATION = (
    "Synthetic research limitation: this offline arithmetic demo uses an uninformative 0.5 risk prior. "
    "The Phase 9.8 pilot uses a separate synthetic, uncalibrated risk signal and counterfactual outcomes. "
    "Neither validates a real-world critic or establishes policy superiority."
)


def run_offline_scenario(
    task_id: str,
    seed: int,
    policy: str,
    trace_path: str | Path = DEFAULT_TRACE_PATH,
    model_path: str | Path = PROJECT_ROOT / DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    """Run one selected task using the existing offline execution function."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task: {task_id}")
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    return run(
        TASK_BY_ID[task_id],
        int(seed),
        Path(trace_path),
        policy,
        Path(model_path),
    )


def planner_output_for_task(task_id: str) -> list[str]:
    """Return the deterministic plan from the same Planner used by ``run``."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task: {task_id}")
    return Planner().plan(TASK_BY_ID[task_id])


def temporary_trace_path(session_id: str, task_id: str, seed: int, policy: str) -> Path:
    """Create a unique trace destination outside frozen project results/logs."""
    if not isinstance(session_id, str) or not session_id or any(c not in "0123456789abcdef-" for c in session_id.lower()):
        raise ValueError("dashboard session identifier is invalid")
    if task_id not in TASK_BY_ID or policy not in (*POLICIES,):
        raise ValueError("unknown task or policy")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    folder = Path(tempfile.gettempdir()) / "safe_intervention_dashboard" / session_id
    return folder / f"{task_id}-{seed}-{policy}-{uuid.uuid4().hex}.jsonl"


def load_metrics_for_dashboard(path: str | Path) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    """Load and recompute metrics from a validated evaluation JSONL log."""
    try:
        summaries = load_run_summaries(path)
        metrics = compute_metrics(summaries)
    except (OSError, ValueError) as exc:
        return None, str(exc)
    return metrics, None


def trace_table_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project existing step events into compact, readable dashboard rows."""
    rows = []
    for event in events:
        verifier = event.get("verifier") or {}
        rows.append({
            "step": event.get("step_index"),
            "failure_prob input": event.get("failure_prob"),
            "decision": event.get("decision"),
            "intervention": bool(event.get("intervention")),
            "replanned": bool(event.get("planner_replanned")),
            "coder instruction": event.get("instruction"),
            "faulted instruction": event.get("faulted_instruction"),
            "code before": event.get("code_before"),
            "code after": event.get("code"),
            "corrective_guidance": event.get("corrective_guidance"),
            "decision_reason": event.get("decision_reason"),
            "step_executed": event.get("step_executed"),
            "verifier_passed": verifier.get("passed"),
            "actual": verifier.get("actual"),
            "expected": verifier.get("expected"),
        })
    return rows


def load_phase98_pilot_results(
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read and validate the frozen Phase 9.8 summary and audit-reported primary estimate.

    The primary estimate is intentionally read from the completed audit report:
    it is not embedded in the frozen comparative metrics JSON. This helper does
    not recalculate any result or read/write scenario records.
    """
    root = Path(project_root)
    run_dir = root / PHASE98_RESULTS_DIR
    metrics_path = run_dir / "comparative_metrics.json"
    pilot_test_path = run_dir / "pilot_test_descriptive_selection.json"
    audit_path = root / PHASE98_AUDIT_PATH
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pilot_test = json.loads(pilot_test_path.read_text(encoding="utf-8"))
        audit = audit_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        return None, "Frozen Phase 9.8 results or audit report are missing or invalid."

    policy_metrics = metrics.get("policy_metrics") if isinstance(metrics, dict) else None
    if (not isinstance(metrics, dict)
            or metrics.get("schema_version") != "phase9-8-pilot-runner-v1"
            or not isinstance(policy_metrics, dict)
            or set(policy_metrics) != set(PHASE98_POLICIES)):
        return None, "Frozen Phase 9.8 metrics do not match the expected policy schema."
    if any(not isinstance(policy_metrics[name], dict)
           or policy_metrics[name].get("attempts") != 2560
           for name in PHASE98_POLICIES):
        return None, "Frozen Phase 9.8 metrics do not contain the expected matched scenario count."
    if (not isinstance(pilot_test, dict)
            or pilot_test.get("policy") != "pilot_test"
            or pilot_test.get("comparative_metric_eligible") is not False):
        return None, "The separate pilot_test record is missing or marked as comparative."

    section = re.search(
        r"### Primary paired cost comparison(.*?)(?=\n## |\Z)", audit, flags=re.DOTALL,
    )
    if section is None:
        return None, "The audit report does not contain the primary comparison section."
    body = section.group(1)
    estimate_match = re.search(r"Estimate:\s*\*\*([-+]?\d+(?:\.\d+)?)\*\*", body)
    interval_match = re.search(
        r"interval:\s*\*\*\[\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\s*\]\*\*",
        body, flags=re.IGNORECASE,
    )
    resamples_match = re.search(r"([\d,]+)\s+resamples", body)
    if not (estimate_match and interval_match and resamples_match):
        return None, "The audited primary estimate or bootstrap details could not be read."
    try:
        primary = {
            "mean_cost_difference": float(estimate_match.group(1)),
            "interval_95": [float(interval_match.group(1)), float(interval_match.group(2))],
            "bootstrap_resamples": int(resamples_match.group(1).replace(",", "")),
            "source": str(PHASE98_AUDIT_PATH).replace("\\", "/"),
            "is_recomputed_from_scored_records": True,
        }
    except ValueError:
        return None, "The audited primary estimate has an invalid numeric format."

    return {
        "policy_metrics": policy_metrics,
        "pilot_test": pilot_test,
        "scenario_count": 2560,
        "matched_decision_count": sum(policy_metrics[name]["attempts"] for name in PHASE98_POLICIES),
        "primary_comparison": primary,
        "metrics_path": str((PHASE98_RESULTS_DIR / "comparative_metrics.json").as_posix()),
        "scored_runs_path": str((PHASE98_RESULTS_DIR / "comparative_scored_runs.jsonl").as_posix()),
        "pilot_test_path": str((PHASE98_RESULTS_DIR / "pilot_test_descriptive_selection.json").as_posix()),
        "disruption_rate_definition": metrics.get("disruption_rate_definition"),
    }, None


def trajectory_chart_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return observed per-step risk and a marker flag for actual interventions."""
    rows = []
    for event in events:
        risk = event.get("failure_prob")
        if isinstance(risk, (int, float)):
            rows.append({
                "step": int(event.get("step_index", len(rows))),
                "failure_prob": float(risk),
                "intervention": event.get("decision") == "intervene",
            })
    return rows


def evaluation_logs(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Return the current held-out offline evaluation trace, never legacy mixed results."""
    root = Path(project_root)
    candidates = [root / "results" / "heldout_evaluation" / "evaluation_runs.jsonl"]
    return [path for path in candidates if path.is_file()]


def downloadable_artifacts(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """List existing offline traces/results; intentionally exclude live logs and secrets."""
    root = Path(project_root)
    candidates = [
        root / "logs" / "dashboard_offline_runs.jsonl",
        root / "logs" / "phase1.jsonl",
        root / "logs" / "phase5_demo.jsonl",
        root / "logs" / "phase7_offline_check.jsonl",
    ]
    results_dir = root / "results"
    if results_dir.is_dir():
        candidates.append(results_dir / "offline_training_examples.jsonl")
        heldout_dir = results_dir / "heldout_evaluation"
        if heldout_dir.is_dir():
            candidates.extend(
                path for path in heldout_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jsonl", ".csv", ".md", ".svg"}
            )
        pilot_dir = root / PHASE98_RESULTS_DIR
        if pilot_dir.is_dir():
            candidates.extend(
                pilot_dir / name for name in (
                    "comparative_metrics.json",
                    "comparative_scored_runs.jsonl",
                    "pilot_test_descriptive_selection.json",
                )
            )
    unique_paths = sorted({path.resolve() for path in candidates if path.is_file()})
    return unique_paths


def metrics_table_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Format actual computed counts/rates with their denominators."""
    rows = []
    for policy in POLICIES:
        item = metrics[policy]
        disruption = (
            "n/a (0 cases)" if item["disruption_rate"] is None
            else f"{item['disruption_count']}/{item['disruption_denominator']} ({item['disruption_rate']:.1%})"
        )
        recovery = (
            "n/a (0 cases)" if item["recovery_rate"] is None
            else f"{item['recovery_count']}/{item['recovery_denominator']} ({item['recovery_rate']:.1%})"
        )
        rows.append({
            "policy": policy,
            "success": f"{item['successes']}/{item['attempts']} ({item['task_success_rate']:.1%})",
            "interventions": item["intervention_count"],
            "intervention_rate": f"{item['intervened_trajectories']}/{item['attempts']} ({item['intervention_rate']:.1%})",
            "disruption": disruption,
            "recovery": recovery,
            "average_executed_steps": item["average_steps"],
            "average_local_latency_seconds": item["average_latency_seconds"],
        })
    return rows
