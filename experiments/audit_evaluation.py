"""Sanity-audit existing policy evaluation logs and rendered reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from experiments.metrics import POLICIES, compute_metrics, load_run_summaries, validate_matched_runs, write_metric_artifacts
from policy.proposed import ProposedPolicy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_repetitions(first_path: str | Path, second_path: str | Path) -> bool:
    """Compare deterministic scenario and outcome fields, excluding IDs and timings."""
    def read(path: str | Path):
        entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        summaries = [entry for entry in entries if entry.get("event_type") == "run_summary"]
        validate_matched_runs(summaries)
        step_map: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            if entry.get("event_type") == "step":
                step_map[entry["run_id"]].append(entry)
        summary_map = {}
        for summary in summaries:
            summary_key = (summary["task_id"], summary["seed"], summary["policy"])
            summary_map[summary_key] = tuple(summary[field] for field in (
                "scenario_hash", "initial_state_hash", "fault_enabled", "passed", "aborted",
                "final_result", "steps", "steps_executed", "intervention_count",
            ))
        stable_steps = {}
        for summary in summaries:
            key = (summary["task_id"], summary["seed"], summary["policy"])
            stable_steps[key] = tuple(
                (step["step_index"], step["decision"], step["failure_prob"], step["features"],
                 step["instruction"], step["faulted_instruction"], step["corrective_guidance"],
                 step["intervention"], step.get("planner_replanned", False), step["step_executed"],
                 step["code"], step["verifier"])
                for step in sorted(step_map[summary["run_id"]], key=lambda item: item["step_index"])
            )
        return summary_map, stable_steps

    return read(first_path) == read(second_path)


def audit_evaluation(
    log_path: str | Path,
    csv_path: str | Path,
    table_path: str | Path,
    svg_path: str | Path,
    model_path: str | Path,
    compare_log_path: str | Path | None = None,
) -> dict:
    log_path, csv_path, table_path, svg_path, model_path = map(Path, (log_path, csv_path, table_path, svg_path, model_path))
    model_hash_before = sha256_file(model_path)
    policy_artifact = ProposedPolicy.load(model_path)
    metadata = policy_artifact.artifact["training_metadata"]
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    summaries = load_run_summaries(log_path)
    validate_matched_runs(summaries)
    metrics = compute_metrics(summaries)
    summaries_by_id = {summary["run_id"]: summary for summary in summaries}
    steps_by_id: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["event_type"] == "step":
            steps_by_id[record["run_id"]].append(record)
    if set(steps_by_id) != set(summaries_by_id):
        raise ValueError("run IDs differ between step records and summaries")

    scenario_keys = {(summary["task_id"], summary["seed"]) for summary in summaries}
    per_policy_outcomes = Counter()
    fault_counts: dict[str, set[bool]] = defaultdict(set)
    proposed_decisions = 0
    proposed_replanned = 0
    recovery_verifier_passes = 0
    for summary in summaries:
        run_steps = steps_by_id[summary["run_id"]]
        if not run_steps:
            raise ValueError(f"run {summary['run_id']} contains no step events")
        if any(step["initial_state"] != {"code": "", "completed_steps": []} for step in run_steps):
            raise ValueError(f"run {summary['run_id']} did not use the expected isolated initial state")
        if any(step["fault_enabled"] != summary["fault_enabled"] for step in run_steps):
            raise ValueError(f"run {summary['run_id']} step fault state differs from summary")
        actual_interventions = [step for step in run_steps if step.get("decision") == "intervene"]
        if len(actual_interventions) != summary["intervention_count"]:
            raise ValueError(f"run {summary['run_id']} intervention count differs from logged decisions")
        for step in actual_interventions:
            if not step.get("intervention") or not step.get("planner_replanned") or not step.get("corrective_guidance"):
                raise ValueError(f"run {summary['run_id']} intervention lacks corrective replanning evidence")
            if step["policy"] == "proposed":
                proposed_decisions += 1
                proposed_replanned += 1
        executed = [step for step in run_steps if step["step_executed"]]
        if summary["passed"]:
            if summary["aborted"] or not executed or not executed[-1].get("verifier", {}).get("passed"):
                raise ValueError(f"run {summary['run_id']} success lacks a final verifier pass")
        elif any(step.get("verifier", {}).get("passed") for step in executed[-1:]):
            raise ValueError(f"run {summary['run_id']} summary failure contradicts final verifier")
        per_policy_outcomes[(summary["policy"], "passed" if summary["passed"] else "failed")] += 1
        fault_counts[summary["policy"]].add(summary["fault_enabled"])

    no_intervention = [s for s in summaries if s["policy"] == "no_intervention"]
    baseline_successes = sum(s["passed"] for s in no_intervention)
    baseline_failures = len(no_intervention) - baseline_successes
    proposed = [s for s in summaries if s["policy"] == "proposed"]
    proposed_interventions = sum(s["intervention_count"] for s in proposed)
    recovery_verifier_passes = sum(
        s["passed"] and s["intervention_count"] > 0
        for s in proposed
        if not next(
            item["passed"] for item in no_intervention
            if item["task_id"] == s["task_id"] and item["seed"] == s["seed"]
        )
    )
    if proposed_decisions != proposed_interventions or proposed_replanned != proposed_interventions:
        raise ValueError("proposed interventions do not all correspond to logged replans")
    if recovery_verifier_passes != metrics["proposed"]["recovery_count"]:
        raise ValueError("recovery numerator differs from verifier-backed matched outcomes")

    # Re-render into a temporary directory and compare byte-for-byte without modifying existing reports.
    with tempfile.TemporaryDirectory(dir=log_path.parent) as temporary:
        expected = write_metric_artifacts(metrics, temporary)
        comparisons = {
            "csv_matches": expected["csv"].read_bytes() == csv_path.read_bytes(),
            "markdown_matches": expected["table"].read_bytes() == table_path.read_bytes(),
            "chart_matches": expected["plot"].read_bytes() == svg_path.read_bytes(),
        }
    if not all(comparisons.values()):
        raise ValueError(f"rendered artifact differs from recomputed raw-log metrics: {comparisons}")
    model_hash_after = sha256_file(model_path)
    if model_hash_before != model_hash_after:
        raise ValueError("saved classifier changed during the read-only audit")
    repeated_match = compare_repetitions(log_path, compare_log_path) if compare_log_path else None
    if compare_log_path and not repeated_match:
        raise ValueError("repeated run differs in scenario, decision, or outcome fields")
    evaluated_tasks = sorted({summary["task_id"] for summary in summaries})
    training_tasks = metadata["training_tasks"]
    declared_evaluation_tasks = metadata["evaluation_tasks"]
    training_task_overlap = sorted(set(evaluated_tasks) & set(training_tasks))
    return {
        "scenarios": len(scenario_keys),
        "runs": len(summaries),
        "policy_counts": dict(Counter(summary["policy"] for summary in summaries)),
        "baseline_successes": baseline_successes,
        "baseline_failures": baseline_failures,
        "fault_states_by_policy": {key: sorted(values) for key, values in fault_counts.items()},
        "proposed_intervention_decisions": proposed_decisions,
        "proposed_replans": proposed_replanned,
        "verifier_backed_recoveries": recovery_verifier_passes,
        "metrics": metrics,
        "report_comparisons": comparisons,
        "repeat_outcome_match": repeated_match,
        "latency_scope": "local execution-loop runtime; excludes model loading and task/scenario setup; not provider/API latency",
        "model_sha256": model_hash_before,
        "model_artifact_version": policy_artifact.artifact["artifact_version"],
        "model_features": list(policy_artifact.artifact["feature_names"]),
        "model_training_tasks": metadata["training_tasks"],
        "model_evaluation_tasks": metadata["evaluation_tasks"],
        "evaluated_tasks": evaluated_tasks,
        "evaluated_training_task_overlap": training_task_overlap,
        "evaluation_uses_only_declared_holdout_tasks": set(evaluated_tasks).issubset(set(declared_evaluation_tasks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("results/evaluation_runs.jsonl"))
    parser.add_argument("--csv", type=Path, default=Path("results/comparison_metrics.csv"))
    parser.add_argument("--table", type=Path, default=Path("results/comparison_summary.md"))
    parser.add_argument("--svg", type=Path, default=Path("results/policy_comparison.svg"))
    parser.add_argument("--model", type=Path, default=Path("models/proposed_policy.joblib"))
    parser.add_argument("--compare-log", type=Path, help="optional repeat evaluation JSONL to compare for deterministic outcomes")
    args = parser.parse_args()
    report = audit_evaluation(args.log, args.csv, args.table, args.svg, args.model, args.compare_log)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
