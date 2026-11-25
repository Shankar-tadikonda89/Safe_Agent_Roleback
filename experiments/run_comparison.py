"""Run matched offline comparisons for the three committed policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from run_demo import TASKS, run
from policy.proposed import DEFAULT_ARTIFACT, ProposedPolicy
from experiments.metrics import (
    POLICIES,
    compute_metrics,
    load_run_summaries,
    validate_matched_runs,
    write_metric_artifacts,
)


def evaluate_suite(
    model_path: str | Path = DEFAULT_ARTIFACT,
    output_dir: str | Path = Path("results"),
    tasks: Iterable | None = None,
    seeds: Iterable[int] = range(32),
) -> dict:
    """Run each policy in a fresh state for every same task/seed scenario."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"saved classifier not found at {model_path}; evaluation does not train or retrain it"
        )
    # Validate model metadata before opening/truncating generated evaluation outputs.
    artifact_policy = ProposedPolicy.load(model_path)
    if tasks is None:
        heldout_ids = artifact_policy.artifact.get("training_metadata", {}).get("evaluation_tasks")
        if not isinstance(heldout_ids, list) or not heldout_ids:
            raise ValueError("saved classifier metadata does not declare held-out evaluation tasks")
        tasks_by_id = {task.task_id: task for task in TASKS}
        unknown = [task_id for task_id in heldout_ids if task_id not in tasks_by_id]
        if unknown:
            raise ValueError(f"artifact evaluation tasks are absent from the task suite: {unknown}")
        tasks = [tasks_by_id[task_id] for task_id in heldout_ids]
    tasks = tuple(tasks)
    seeds = tuple(seeds)
    if not tasks or not seeds:
        raise ValueError("evaluation requires at least one task and one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seed list contains duplicates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "evaluation_runs.jsonl"
    # This file is a generated result artifact; start each evaluation from a clean trace.
    trace_path.write_text("", encoding="utf-8")

    expected = len(tasks) * len(seeds) * len(POLICIES)
    completed = 0
    for task in tasks:
        for seed in seeds:
            # Every run() call constructs its own equivalent scenario and empty initial state.
            for policy_name in POLICIES:
                run(task, seed, trace_path, policy_name, model_path)
                completed += 1
    summaries = load_run_summaries(trace_path)
    if len(summaries) != expected or completed != expected:
        raise ValueError(f"expected {expected} run summaries, found {len(summaries)}")
    validate_matched_runs(summaries)
    metrics = compute_metrics(summaries)
    artifacts = write_metric_artifacts(metrics, output_dir)
    return {
        "run_count": len(summaries),
        "scenario_count": len(tasks) * len(seeds),
        "task_ids": [task.task_id for task in tasks],
        "seeds": list(seeds),
        "metrics": metrics,
        "trace": trace_path,
        **artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--tasks", default="heldout", help="heldout (artifact-declared) or all or comma-separated task IDs")
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.tasks == "heldout":
        tasks = None
    elif args.tasks == "all":
        tasks = TASKS
    else:
        by_id = {task.task_id: task for task in TASKS}
        task_ids = [part.strip() for part in args.tasks.split(",") if part.strip()]
        if not task_ids or any(task_id not in by_id for task_id in task_ids):
            parser.error("--tasks must be 'all' or a comma-separated list of valid task IDs")
        tasks = [by_id[task_id] for task_id in task_ids]
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    result = evaluate_suite(args.model, args.output_dir, tasks, seeds)
    print(f"Matched scenarios: {result['scenario_count']} ({len(result['task_ids'])} tasks x {len(result['seeds'])} seeds)")
    print(f"Policy runs: {result['run_count']} ({', '.join(POLICIES)})")
    print(f"Run log: {result['trace']}")
    print(f"Metrics: {result['table']}")
    print(f"CSV: {result['csv']}")
    print(f"Plot: {result['plot']}")
    print()
    print(result["table"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
