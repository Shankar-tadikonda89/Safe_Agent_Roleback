"""Strict JSONL validation and matched three-policy metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


POLICIES = ("no_intervention", "naive_threshold", "proposed")
SUMMARY_REQUIRED = (
    "run_id", "policy", "task_id", "seed", "scenario_hash", "initial_state_hash",
    "fault_enabled", "passed", "aborted", "steps", "steps_executed", "intervention_count",
    "latency_seconds",
)
STEP_REQUIRED = (
    "run_id", "policy", "task_id", "seed", "scenario_hash", "initial_state_hash",
    "branch_type", "decision", "step_executed",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_summary(entry: dict[str, Any], line_no: int) -> None:
    missing = [key for key in SUMMARY_REQUIRED if key not in entry]
    _require(not missing, f"line {line_no}: run_summary missing fields: {', '.join(missing)}")
    _require(entry["policy"] in POLICIES, f"line {line_no}: unknown policy {entry['policy']!r}")
    _require(isinstance(entry["run_id"], str) and bool(entry["run_id"]), f"line {line_no}: invalid run_id")
    _require(isinstance(entry["task_id"], str) and bool(entry["task_id"]), f"line {line_no}: invalid task_id")
    _require(isinstance(entry["seed"], int) and not isinstance(entry["seed"], bool), f"line {line_no}: invalid seed")
    _require(isinstance(entry["passed"], bool), f"line {line_no}: passed must be boolean")
    _require(isinstance(entry["aborted"], bool), f"line {line_no}: aborted must be boolean")
    _require(not (entry["aborted"] and entry["passed"]), f"line {line_no}: aborted run cannot pass")
    _require(isinstance(entry["fault_enabled"], bool), f"line {line_no}: fault_enabled must be boolean")
    for key in ("scenario_hash", "initial_state_hash"):
        _require(isinstance(entry[key], str) and bool(entry[key]), f"line {line_no}: invalid {key}")
    for key in ("steps_executed", "intervention_count"):
        _require(isinstance(entry[key], int) and not isinstance(entry[key], bool) and entry[key] >= 0,
                 f"line {line_no}: {key} must be a non-negative integer")
    latency = entry["latency_seconds"]
    _require(isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0,
             f"line {line_no}: invalid latency_seconds")


def load_run_summaries(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate run summaries from a step-and-summary JSONL trace."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"JSONL log does not exist: {path}")
    summaries: list[dict[str, Any]] = []
    steps_by_run: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"line {line_no}: blank JSONL record")
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"line {line_no}: JSONL record must be an object")
            event_type = entry.get("event_type")
            if event_type == "run_summary":
                _validate_summary(entry, line_no)
                summaries.append(entry)
            elif event_type == "step":
                missing = [key for key in STEP_REQUIRED if key not in entry]
                _require(not missing, f"line {line_no}: step missing fields: {', '.join(missing)}")
                _require(entry["branch_type"] == "live", f"line {line_no}: evaluation step must be live")
                _require(entry["policy"] in POLICIES, f"line {line_no}: unknown step policy")
                _require(isinstance(entry["step_executed"], bool), f"line {line_no}: step_executed must be boolean")
                _require(entry["decision"] in {"continue", "intervene", "abort"}, f"line {line_no}: invalid decision")
                steps_by_run.setdefault(entry["run_id"], []).append(entry)
            else:
                raise ValueError(f"line {line_no}: unsupported event_type {event_type!r}")
    _require(bool(summaries), "JSONL log contains no run_summary records")
    ids = [summary["run_id"] for summary in summaries]
    _require(len(ids) == len(set(ids)), "duplicate run_id in summaries")
    summaries_by_id = {summary["run_id"]: summary for summary in summaries}
    _require(set(steps_by_run) == set(summaries_by_id), "step records and run summaries have mismatched run IDs")
    for run_id, records in steps_by_run.items():
        summary = summaries_by_id[run_id]
        _require(len(records) == summary["steps"], f"run {run_id}: steps count does not match summary")
        _require(sum(record["step_executed"] for record in records) == summary["steps_executed"],
                 f"run {run_id}: executed step count does not match summary")
        for record in records:
            for field in ("policy", "task_id", "seed", "scenario_hash", "initial_state_hash"):
                _require(record[field] == summary[field], f"run {run_id}: step/summary mismatch in {field}")
    return summaries


def validate_matched_runs(summaries: Iterable[dict[str, Any]]) -> None:
    summaries = list(summaries)
    by_scenario: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for entry in summaries:
        key = (entry["task_id"], entry["seed"])
        policies = by_scenario.setdefault(key, {})
        _require(entry["policy"] not in policies, f"duplicate policy run for {key}: {entry['policy']}")
        policies[entry["policy"]] = entry
    _require(bool(by_scenario), "no matched scenarios in evaluation data")
    for key, policies in by_scenario.items():
        _require(set(policies) == set(POLICIES), f"scenario {key} missing or has extra policies")
        reference = policies[POLICIES[0]]
        for policy in POLICIES[1:]:
            candidate = policies[policy]
            for field in ("scenario_hash", "initial_state_hash", "fault_enabled"):
                _require(candidate[field] == reference[field], f"scenario {key} has mismatched {field}")


def compute_metrics(summaries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-policy rates with explicit numerator and denominator counts.

    Disruption denominator: matched cases whose no_intervention run passed. An
    intervention on any such case counts as disruption because it changes the
    trajectory despite the control already passing.
    Recovery denominator: matched cases whose no_intervention run failed.
    Recovery numerator: the policy intervened and its verifier passed.
    Average steps: executed steps / attempted trajectories; an abort executes 0.
    """
    summaries = list(summaries)
    validate_matched_runs(summaries)
    by_key = {(item["task_id"], item["seed"], item["policy"]): item for item in summaries}
    keys = sorted({(item["task_id"], item["seed"]) for item in summaries})
    control_success = [key for key in keys if by_key[(*key, "no_intervention")]["passed"]]
    control_failure = [key for key in keys if not by_key[(*key, "no_intervention")]["passed"]]
    output: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        runs = [by_key[(*key, policy)] for key in keys]
        successes = sum(item["passed"] for item in runs)
        interventions = sum(item["intervention_count"] for item in runs)
        intervened_runs = sum(item["intervention_count"] > 0 for item in runs)
        disruptions = sum(by_key[(*key, policy)]["intervention_count"] > 0 for key in control_success)
        recoveries = sum(
            by_key[(*key, policy)]["passed"] and by_key[(*key, policy)]["intervention_count"] > 0
            for key in control_failure
        )
        executed_steps = sum(item["steps_executed"] for item in runs)
        output[policy] = {
            "attempts": len(runs),
            "successes": successes,
            "task_success_rate": successes / len(runs),
            "intervention_count": interventions,
            "intervened_trajectories": intervened_runs,
            "intervention_rate": intervened_runs / len(runs),
            "disruption_count": disruptions,
            "disruption_denominator": len(control_success),
            "disruption_rate": disruptions / len(control_success) if control_success else None,
            "recovery_count": recoveries,
            "recovery_denominator": len(control_failure),
            "recovery_rate": recoveries / len(control_failure) if control_failure else None,
            "executed_steps": executed_steps,
            "average_steps": executed_steps / len(runs),
            "average_latency_seconds": sum(item["latency_seconds"] for item in runs) / len(runs),
        }
    return output


def write_metric_artifacts(metrics: dict[str, dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Write the comparison table, CSV and dependency-free SVG from log metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "comparison_summary.md"
    csv_path = output_dir / "comparison_metrics.csv"
    svg_path = output_dir / "policy_comparison.svg"
    with table_path.open("w", encoding="utf-8") as stream:
        stream.write("# Policy comparison (computed from JSONL run summaries)\n\n")
        stream.write("Rates include numerator/denominator. Disruption uses no-intervention successes as its denominator; recovery uses no-intervention failures. Aborted runs execute zero steps.\n\n")
        stream.write("| Policy | Success | Interventions | Intervention rate | Disruption | Recovery | Avg executed steps | Avg latency (s) |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for policy in POLICIES:
            item = metrics[policy]
            disruption = "n/a (0)" if item["disruption_rate"] is None else f"{item['disruption_count']}/{item['disruption_denominator']} ({item['disruption_rate']:.1%})"
            recovery = "n/a (0)" if item["recovery_rate"] is None else f"{item['recovery_count']}/{item['recovery_denominator']} ({item['recovery_rate']:.1%})"
            stream.write(
                f"| {policy} | {item['successes']}/{item['attempts']} ({item['task_success_rate']:.1%}) "
                f"| {item['intervention_count']} | {item['intervened_trajectories']}/{item['attempts']} ({item['intervention_rate']:.1%}) "
                f"| {disruption} | {recovery} | {item['average_steps']:.3f} | {item['average_latency_seconds']:.6f} |\n"
            )
    columns = list(next(iter(metrics.values())).keys())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", *columns])
        writer.writeheader()
        for policy in POLICIES:
            writer.writerow({"policy": policy, **metrics[policy]})
    _write_svg(metrics, svg_path)
    return {"table": table_path, "csv": csv_path, "plot": svg_path}


def _write_svg(metrics: dict[str, dict[str, Any]], path: Path) -> None:
    rate_keys = ("task_success_rate", "intervention_rate", "disruption_rate", "recovery_rate")
    labels = ("Success", "Intervention", "Disruption", "Recovery")
    colors = ("#3b82f6", "#f59e0b", "#ef4444")
    width, height = 900, 480
    left, top, chart_width, chart_height = 95, 80, 750, 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="35" text-anchor="middle" font-family="sans-serif" font-size="20">Policy comparison from recorded runs</text>',
    ]
    for tick in range(0, 11, 2):
        y = top + chart_height - chart_height * tick / 10
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick * 10}%</text>')
    group_width = chart_width / len(rate_keys)
    bar_width = 38
    for group, (metric_key, label) in enumerate(zip(rate_keys, labels)):
        center = left + group_width * (group + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top + chart_height + 28}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>')
        for index, policy in enumerate(POLICIES):
            value = metrics[policy][metric_key]
            x = center + (index - 1) * (bar_width + 8) - bar_width / 2
            if value is None:
                parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height - 8}" text-anchor="middle" font-family="sans-serif" font-size="10">n/a</text>')
                continue
            bar_height = max(0.0, min(1.0, float(value))) * chart_height
            y = top + chart_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index]}"/>')
            parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{max(top - 2, y - 5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{float(value):.0%}</text>')
    legend_y = 435
    for index, policy in enumerate(POLICIES):
        x = 250 + index * 190
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[index]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-family="sans-serif" font-size="12">{policy}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
