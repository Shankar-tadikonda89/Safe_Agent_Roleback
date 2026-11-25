import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import helpers
from experiments.metrics import POLICIES
from run_demo import TASKS


class DashboardHelperTests(unittest.TestCase):
    def test_offline_run_delegates_to_existing_runner_with_selected_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            trace = Path(directory) / "trace.jsonl"
            model = Path(directory) / "model.joblib"
            expected = {"events": [], "summary": {"passed": True}}
            with patch("dashboard.helpers.run", return_value=expected) as runner:
                result = helpers.run_offline_scenario("task_03", 19, "no_intervention", trace, model)
            self.assertIs(result, expected)
            args = runner.call_args.args
            self.assertEqual(args[0], next(task for task in TASKS if task.task_id == "task_03"))
            self.assertEqual(args[1:4], (19, trace, "no_intervention"))
            self.assertEqual(args[4], model)
            with self.assertRaises(ValueError):
                helpers.run_offline_scenario("unknown", 19, "no_intervention", trace, model)
            with self.assertRaises(ValueError):
                helpers.run_offline_scenario("task_03", 19, "live", trace, model)

    def test_trace_and_chart_projection_preserve_actual_intervention_and_verifier(self):
        events = [{
            "step_index": 0,
            "failure_prob": 0.8,
            "decision": "intervene",
            "intervention": True,
            "planner_replanned": True,
            "corrective_guidance": "Use the canonical instruction.",
            "step_executed": True,
            "verifier": {"passed": True, "actual": 42, "expected": 42},
        }]
        trace = helpers.trace_table_rows(events)
        self.assertEqual(trace[0]["decision"], "intervene")
        self.assertTrue(trace[0]["replanned"])
        self.assertEqual(trace[0]["verifier_passed"], True)
        self.assertEqual(trace[0]["actual"], 42)
        self.assertEqual(helpers.trajectory_chart_rows(events), [
            {"step": 0, "failure_prob": 0.8, "intervention": True},
        ])

    def test_metric_loader_recomputes_from_valid_jsonl_and_handles_missing_or_empty(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            path = root / "evaluation.jsonl"
            entries = []
            for policy in POLICIES:
                run_id = f"run-{policy}"
                aborted = policy == "naive_threshold"
                passed = policy != "naive_threshold"
                entries.append({
                    "event_type": "step", "run_id": run_id, "policy": policy,
                    "task_id": "task_fixture", "seed": 1, "scenario_hash": "scenario",
                    "initial_state_hash": "initial", "branch_type": "live",
                    "decision": "abort" if aborted else "continue", "step_executed": not aborted,
                })
                entries.append({
                    "event_type": "run_summary", "run_id": run_id, "policy": policy,
                    "task_id": "task_fixture", "seed": 1, "scenario_hash": "scenario",
                    "initial_state_hash": "initial", "fault_enabled": True, "passed": passed,
                    "aborted": aborted, "steps": 1, "steps_executed": 0 if aborted else 1,
                    "intervention_count": 1 if policy == "proposed" else 0,
                    "latency_seconds": 0.01,
                })
            path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            metrics, error = helpers.load_metrics_for_dashboard(path)
            self.assertIsNone(error)
            self.assertEqual(metrics["no_intervention"]["task_success_rate"], 1.0)
            self.assertEqual(metrics["naive_threshold"]["average_steps"], 0.0)
            rows = helpers.metrics_table_rows(metrics)
            self.assertEqual([row["policy"] for row in rows], list(POLICIES))
            self.assertIn("1/1", rows[0]["success"])

            missing, missing_error = helpers.load_metrics_for_dashboard(root / "missing.jsonl")
            self.assertIsNone(missing)
            self.assertTrue(missing_error)
            empty_path = root / "empty.jsonl"
            empty_path.write_text("", encoding="utf-8")
            empty, empty_error = helpers.load_metrics_for_dashboard(empty_path)
            self.assertIsNone(empty)
            self.assertTrue(empty_error)

    def test_downloads_are_limited_to_offline_logs_and_result_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            heldout = root / "results" / "heldout_evaluation"
            heldout.mkdir(parents=True)
            (root / "logs" / "dashboard_offline_runs.jsonl").write_text("offline\n", encoding="utf-8")
            (root / "logs" / "live_runs.jsonl").write_text("live\n", encoding="utf-8")
            (heldout / "comparison_metrics.csv").write_text("policy\n", encoding="utf-8")
            files = helpers.downloadable_artifacts(root)
            names = {path.name for path in files}
            self.assertIn("dashboard_offline_runs.jsonl", names)
            self.assertIn("comparison_metrics.csv", names)
            self.assertNotIn("live_runs.jsonl", names)

    def test_evaluation_log_discovery_prefers_only_existing_known_offline_logs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            heldout = root / "results" / "heldout_evaluation" / "evaluation_runs.jsonl"
            heldout.parent.mkdir(parents=True)
            heldout.write_text("", encoding="utf-8")
            found = helpers.evaluation_logs(root)
            self.assertEqual(found, [heldout])

    def test_dashboard_app_has_valid_python_syntax_without_streamlit_import(self):
        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        source = app_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(app_path))
        self.assertIn("with st.form(\"offline_execution_form\"", source)
        self.assertIn("if run_clicked:", source)
        self.assertIn("dashboard_last_result", source)

    def test_temporary_trace_paths_are_unique_and_outside_frozen_results(self):
        first = helpers.temporary_trace_path("abc123", "task_02", 42, "proposed")
        second = helpers.temporary_trace_path("abc123", "task_02", 42, "proposed")
        self.assertNotEqual(first, second)
        self.assertEqual(first.suffix, ".jsonl")
        self.assertFalse(str(first).startswith(str(helpers.PROJECT_ROOT / "results")))
        with self.assertRaises(ValueError):
            helpers.temporary_trace_path("abc123", "task_02", -1, "proposed")

    def test_actual_offline_runner_integration_writes_only_to_supplied_temp_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "actual-run.jsonl"
            result = helpers.run_offline_scenario("task_02", 42, "no_intervention", trace)
            self.assertTrue(result["summary"]["passed"])
            self.assertEqual(result["summary"]["llm_calls"], 0)
            self.assertTrue(trace.is_file())
            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["event_type"], "run_summary")
            self.assertEqual(records[-1]["policy"], "no_intervention")

    def test_planner_output_comes_from_existing_planner(self):
        from run_demo import Planner, TASK_BY_ID

        self.assertEqual(
            helpers.planner_output_for_task("task_01"),
            Planner().plan(TASK_BY_ID["task_01"]),
        )

    @staticmethod
    def _phase98_fixture(root: Path) -> None:
        run_dir = root / helpers.PHASE98_RESULTS_DIR
        run_dir.mkdir(parents=True)
        policies = {}
        for policy in helpers.PHASE98_POLICIES:
            policies[policy] = {
                "attempts": 2560,
                "success_count": 0 if policy == "naive_threshold" else 1,
                "task_success_rate": 0.5,
                "mean_realized_task_cost": 1.0,
                "intervention_count": 232 if policy == "phase9_8_native_prefix_forecast_v1" else 0,
                "confirmed_application_count": 0,
            }
        (run_dir / "comparative_metrics.json").write_text(json.dumps({
            "schema_version": "phase9-8-pilot-runner-v1",
            "policy_metrics": policies,
            "disruption_rate_definition": "selected counterfactual actions / attempts",
        }), encoding="utf-8")
        (run_dir / "pilot_test_descriptive_selection.json").write_text(json.dumps({
            "policy": "pilot_test", "selected_action": "continue", "scenarios_used": 2560,
            "comparative_metric_eligible": False,
        }), encoding="utf-8")
        audit = root / helpers.PHASE98_AUDIT_PATH
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(
            "### Primary paired cost comparison\n"
            "- Estimate: **0.000000** over 2,560 matched scenarios.\n"
            "- Paired mechanism-stratified percentile interval: **[0.000000, 0.000000]**.\n"
            "- Bootstrap: 10,000 resamples.\n",
            encoding="utf-8",
        )

    def test_frozen_phase98_results_load_from_metrics_and_audit_without_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._phase98_fixture(root)
            result, error = helpers.load_phase98_pilot_results(root)
        self.assertIsNone(error)
        self.assertEqual(result["scenario_count"], 2560)
        self.assertEqual(result["matched_decision_count"], 10240)
        self.assertEqual(result["primary_comparison"]["mean_cost_difference"], 0.0)
        self.assertEqual(result["primary_comparison"]["interval_95"], [0.0, 0.0])
        self.assertEqual(result["primary_comparison"]["bootstrap_resamples"], 10000)
        self.assertFalse(result["pilot_test"]["comparative_metric_eligible"])

    def test_frozen_phase98_loader_handles_missing_invalid_or_inconsistent_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing, missing_error = helpers.load_phase98_pilot_results(root)
            self.assertIsNone(missing)
            self.assertTrue(missing_error)

            self._phase98_fixture(root)
            metrics_path = root / helpers.PHASE98_RESULTS_DIR / "comparative_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["policy_metrics"].pop("proposed")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            invalid, invalid_error = helpers.load_phase98_pilot_results(root)
            self.assertIsNone(invalid)
            self.assertTrue(invalid_error)

    def test_downloads_include_phase98_records_but_never_live_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / helpers.PHASE98_RESULTS_DIR
            run_dir.mkdir(parents=True)
            for name in ("comparative_metrics.json", "comparative_scored_runs.jsonl",
                         "pilot_test_descriptive_selection.json"):
                (run_dir / name).write_text("{}", encoding="utf-8")
            (root / "logs").mkdir()
            (root / "logs" / "live_runs.jsonl").write_text("secret-free fixture", encoding="utf-8")
            names = {path.name for path in helpers.downloadable_artifacts(root)}
        self.assertTrue({"comparative_metrics.json", "comparative_scored_runs.jsonl",
                         "pilot_test_descriptive_selection.json"}.issubset(names))
        self.assertNotIn("live_runs.jsonl", names)


if __name__ == "__main__":
    unittest.main()
