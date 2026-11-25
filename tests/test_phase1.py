import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import joblib

from run_demo import (
    Coder,
    FEATURE_NAMES,
    NaiveThresholdPolicy,
    NoInterventionPolicy,
    TASK,
    TASKS,
    estimate_failure_probability,
    extract_features,
    features_from_observation,
    make_scenario,
    run,
)
from policy.proposed import ProposedPolicy
from policy.train_policy import generate_dataset, make_example, train_and_save
from policy import predict_policy
from experiments.metrics import compute_metrics, load_run_summaries, validate_matched_runs
from experiments.run_comparison import evaluate_suite
from experiments.audit_evaluation import compare_repetitions


class PhaseOneTests(unittest.TestCase):
    def test_same_seed_reproduces_scenario(self):
        self.assertEqual(make_scenario(TASK, 42), make_scenario(TASK, 42))
        self.assertNotEqual(make_scenario(TASK, 42).scenario_hash, make_scenario(TASK, 43).scenario_hash)

    def test_task_suite_validates_and_has_faulted_and_clean_cases(self):
        self.assertEqual(len(TASKS), 8)
        self.assertEqual(len({task.task_id for task in TASKS}), len(TASKS))
        states = {make_scenario(task, 42).fault_enabled for task in TASKS}
        self.assertEqual(states, {True, False})

    def test_coder_only_receives_step_and_accumulated_code(self):
        instruction = "Return the integer result 40."
        code = Coder().implement(instruction, "")
        self.assertEqual(code, "answer = 40\n")
        self.assertNotIn(TASK.prompt, instruction)

    def test_seeded_fault_fails_oracle_and_writes_valid_trace(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            trace = Path(directory) / "trace.jsonl"
            result = run(TASK, 42, trace)
            self.assertFalse(result["summary"]["passed"])
            self.assertNotEqual(result["summary"]["final_result"], TASK.expected)
            entries = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["event_type"] for item in entries], ["step", "run_summary"])
            self.assertTrue(all(item["run_id"] for item in entries))
            self.assertEqual(entries[0]["branch_type"], "live")
            self.assertFalse(entries[-1]["passed"])

    def test_non_faulted_scenario_passes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            result = run(TASKS[1], 42, Path(directory) / "trace.jsonl")
            self.assertFalse(result["scenario"].fault_enabled)
            self.assertTrue(result["summary"]["passed"])
            self.assertEqual(result["summary"]["final_result"], TASKS[1].expected)


class PhaseThreeTests(unittest.TestCase):
    def test_features_are_named_finite_and_in_range(self):
        risk = estimate_failure_probability("Compute 19 + 23 and assign the answer.", "")
        features = extract_features(risk, "code_assignment", 0, 1)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["failure_prob"], 0.5)
        self.assertEqual(features["reversibility_proxy"], 0.25)
        self.assertEqual(features["trajectory_position"], 0.0)
        self.assertEqual(features["disruption_history"], 0.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features.values()))

    def test_features_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            extract_features(float("nan"), "code_assignment", 0, 1)
        with self.assertRaises(ValueError):
            extract_features(0.4, "unknown", 0, 1)
        with self.assertRaises(ValueError):
            extract_features(0.4, "code_assignment", 2, 1)

    def test_no_intervention_always_continues_and_ignores_risk(self):
        policy = NoInterventionPolicy()
        for risk in (0.0, 0.5, 0.500001, 1.0):
            self.assertEqual(policy.decide(risk, {}), "continue")

    def test_naive_threshold_aborts_only_above_point_five(self):
        policy = NaiveThresholdPolicy()
        for risk in (0.0, 0.49, 0.5):
            self.assertEqual(policy.decide(risk, {}), "continue")
        for risk in (0.500001, 0.8, 1.0):
            self.assertEqual(policy.decide(risk, {}), "abort")

    def test_policies_and_risk_are_deterministic_for_seed(self):
        self.assertEqual(
            estimate_failure_probability("same observation", ""),
            estimate_failure_probability("same observation", ""),
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            first = run(TASK, 42, Path(directory) / "a.jsonl", "naive_threshold")
            second = run(TASK, 42, Path(directory) / "b.jsonl", "naive_threshold")
            self.assertEqual(first["summary"]["scenario_hash"], second["summary"]["scenario_hash"])
            self.assertEqual(first["events"][0]["failure_prob"], second["events"][0]["failure_prob"])
            self.assertEqual(first["events"][0]["features"], second["events"][0]["features"])

    def test_neutral_prior_keeps_threshold_baseline_running_without_fixture_leak(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            result = run(TASK, 42, Path(directory) / "trace.jsonl", "naive_threshold")
            event = result["events"][0]
            self.assertEqual(event["failure_prob"], 0.5)
            self.assertEqual(event["decision"], "continue")
            self.assertFalse(result["summary"]["aborted"])
            self.assertFalse(result["summary"]["passed"])
            self.assertFalse(event["verifier"]["passed"])

    def test_fault_flag_alone_cannot_change_risk_or_become_a_feature(self):
        observed_instruction = "The same current instruction is visible in both controlled cases."
        observed_code = ""
        faulted = make_scenario(TASK, 42)
        clean = make_scenario(TASKS[1], 42)
        self.assertNotEqual(faulted.fault_enabled, clean.fault_enabled)
        cases = {
            scenario.fault_enabled: features_from_observation(
                observed_instruction, observed_code, "code_assignment", 0, 1
            )
            for scenario in (faulted, clean)
        }
        self.assertEqual(cases[True], cases[False])
        features = cases[True]
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertFalse({"fault_enabled", "passed", "expected", "actual"} & set(features))


class PhaseFourTests(unittest.TestCase):
    def test_offline_branches_match_prefix_and_labels_follow_oracle_outcomes(self):
        examples = generate_dataset(seeds=range(4))
        self.assertTrue(examples)
        for example in examples:
            continuation = example["continue_branch"]
            intervention = example["intervene_branch"]
            self.assertEqual(continuation["branch_type"], "offline_counterfactual")
            self.assertEqual(intervention["branch_type"], "offline_counterfactual")
            self.assertEqual(continuation["scenario_hash"], intervention["scenario_hash"])
            self.assertEqual(continuation["initial_state"], intervention["initial_state"])
            self.assertEqual(example["intervene_helped"], int(intervention["passed"] and not continuation["passed"]))
            self.assertEqual(example["continue_cost"], 3.0 if example["intervene_helped"] else 0.0)
            self.assertEqual(example["intervene_cost"], 1.0 if example["intervene_helped"] else 2.0)
            self.assertEqual(tuple(example["features"]), FEATURE_NAMES)
            self.assertEqual(
                example["features"],
                features_from_observation(continuation["instruction"], "", "code_assignment", 0, 1),
            )

    def test_training_is_reproducible_and_tasks_are_group_separated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            first = train_and_save(root / "first.joblib", root / "first.jsonl", seeds=range(4))
            second = train_and_save(root / "second.joblib", root / "second.jsonl", seeds=range(4))
            first_meta = first["artifact"]["training_metadata"]
            second_meta = second["artifact"]["training_metadata"]
            self.assertEqual(first["examples"], second["examples"])
            self.assertTrue(np.array_equal(
                first["artifact"]["model"].named_steps["classifier"].coef_,
                second["artifact"]["model"].named_steps["classifier"].coef_,
            ))
            self.assertFalse(set(first_meta["training_tasks"]) & set(first_meta["evaluation_tasks"]))
            self.assertEqual(first_meta["training_tasks"], second_meta["training_tasks"])
            self.assertEqual(first_meta["evaluation_tasks"], second_meta["evaluation_tasks"])
            self.assertGreater(first_meta["train_rows"], 0)
            self.assertGreater(first_meta["evaluation_rows"], 0)
            self.assertEqual(first_meta["split_unit"], "task_id")

    def test_artifact_save_load_feature_order_and_single_prediction_shape(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            dataset_path = Path(directory) / "data.jsonl"
            train_and_save(artifact_path, dataset_path, seeds=range(4))
            policy = ProposedPolicy.load(artifact_path)
            features = features_from_observation(
                "Return the integer result 40.", "", "code_assignment", 0, 1
            )
            with patch.object(policy.model, "predict_proba", wraps=policy.model.predict_proba) as predict:
                result = policy.decide(features)
            self.assertEqual(predict.call_count, 1)
            self.assertEqual(predict.call_args.args[0].shape, (1, len(FEATURE_NAMES)))
            self.assertEqual(result["decision"], "continue")
            self.assertTrue(0.0 <= result["helpful_intervention_probability"] <= 1.0)
            self.assertAlmostEqual(result["expected_intervene_cost"], result["expected_continue_cost"])
            with self.assertRaises(ValueError):
                policy.decide(dict(reversed(list(features.items()))))

    def test_proposed_inference_does_not_call_offline_branching(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            train_and_save(artifact_path, Path(directory) / "data.jsonl", seeds=range(4))
            policy = ProposedPolicy.load(artifact_path)
            features = extract_features(0.8, "code_assignment", 0, 1)
            with patch("policy.train_policy.simulate_counterfactual_branch", side_effect=AssertionError("branch called")) as branch:
                decision = policy.decide(features)
            self.assertIn(decision["decision"], {"continue", "intervene"})
            branch.assert_not_called()

    def test_proposed_classifier_has_no_decision_signal_from_neutral_features(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            trained = train_and_save(artifact_path, Path(directory) / "data.jsonl", seeds=range(4))
            policy = ProposedPolicy.load(artifact_path)
            features = features_from_observation(
                "Return the integer result 40.", "", "code_assignment", 0, 1
            )
            result = policy.decide(features)
            self.assertEqual(result["decision"], "continue")
            self.assertAlmostEqual(result["helpful_intervention_probability"], 0.5)
            self.assertEqual(
                trained["artifact"]["feature_names"],
                list(FEATURE_NAMES),
            )

    def test_prediction_command_uses_the_same_feature_path_as_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            train_and_save(artifact_path, Path(directory) / "data.jsonl", seeds=range(4))
            captured = {}

            def capture(features):
                captured.update(features)
                return {
                    "decision": "continue", "helpful_intervention_probability": 0.5,
                    "expected_continue_cost": 1.5, "expected_intervene_cost": 1.5,
                }

            expected = make_example(TASK, 42)["features"]
            with patch("policy.predict_policy.ProposedPolicy.decide", side_effect=capture), \
                    patch("sys.argv", ["predict_policy", "--task", "task_01", "--seed", "42", "--artifact", str(artifact_path)]):
                self.assertEqual(predict_policy.main(), 0)
            self.assertEqual(captured, expected)
            self.assertEqual(tuple(captured), FEATURE_NAMES)

    def test_invalid_artifact_schema_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            train_and_save(artifact_path, Path(directory) / "data.jsonl", seeds=range(4))
            artifact = joblib.load(artifact_path)
            artifact["feature_names"] = list(reversed(artifact["feature_names"]))
            with self.assertRaises(ValueError):
                ProposedPolicy(artifact)

    def test_pre_correction_artifact_version_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            artifact_path = Path(directory) / "model.joblib"
            train_and_save(artifact_path, Path(directory) / "data.jsonl", seeds=range(4))
            artifact = joblib.load(artifact_path)
            artifact["artifact_version"] = 1
            artifact["training_metadata"].pop("risk_signal")
            with self.assertRaisesRegex(ValueError, "unsupported policy artifact version"):
                ProposedPolicy(artifact)


class PhaseFiveTests(unittest.TestCase):
    def test_proposed_intervention_repairs_faulted_task_from_current_step(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            train_and_save(model_path, root / "data.jsonl", seeds=range(4))
            prediction = {
                "decision": "intervene", "helpful_intervention_probability": 0.9,
                "expected_continue_cost": 2.7, "expected_intervene_cost": 1.1,
            }
            with patch("policy.proposed.ProposedPolicy.decide", return_value=prediction), \
                    patch("policy.train_policy.simulate_counterfactual_branch", side_effect=AssertionError("live branch called")) as branch:
                result = run(TASK, 42, root / "run.jsonl", "proposed", model_path)
            event = result["events"][0]
            self.assertTrue(result["summary"]["passed"])
            self.assertEqual(result["summary"]["final_result"], TASK.expected)
            self.assertEqual(event["decision"], "intervene")
            self.assertTrue(event["intervention"])
            self.assertTrue(event["planner_replanned"])
            self.assertIn("Ignore any corrupted instruction", event["corrective_guidance"])
            self.assertEqual(event["faulted_instruction"], "Return the integer result 40.")
            self.assertEqual(event["verifier"]["actual"], TASK.expected)
            self.assertEqual(event["branch_type"], "live")
            branch.assert_not_called()

    def test_replan_preserves_accumulated_code_and_prior_intervention_history(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            train_and_save(model_path, root / "data.jsonl", seeds=range(4))
            prediction = {
                "decision": "intervene",
                "helpful_intervention_probability": 0.9,
                "expected_continue_cost": 2.7,
                "expected_intervene_cost": 1.1,
            }
            with patch("run_demo.Planner.plan", return_value=[
                "Compute 19 + 23; store the result in a variable named answer.",
                "Compute 19 + 23; store the result in a variable named answer.",
            ]), patch("policy.proposed.ProposedPolicy.decide", side_effect=[
                prediction, {**prediction, "decision": "continue"},
            ]):
                result = run(TASK, 42, root / "run.jsonl", "proposed", model_path)
            first, second = result["events"]
            self.assertEqual(first["decision"], "intervene")
            self.assertEqual(second["decision"], "continue")
            self.assertEqual(first["code_before"], "")
            self.assertEqual(second["code_before"], first["code"])
            self.assertTrue(second["code"].startswith(first["code"]))
            self.assertEqual(second["features"]["disruption_history"], 1.0)
            self.assertEqual(result["summary"]["intervention_count"], 1)
            self.assertTrue(result["summary"]["passed"])

    def test_proposed_policy_does_not_change_existing_no_intervention_result(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            result = run(TASK, 42, Path(directory) / "no_intervention.jsonl")
            self.assertFalse(result["summary"]["passed"])
            self.assertEqual(result["summary"]["final_result"], 40)
            self.assertEqual(result["summary"]["intervention_count"], 0)


class PhaseSixTests(unittest.TestCase):
    @staticmethod
    def summary(run_id, policy, task_id, seed, passed, aborted=False, steps=1, executed=1, interventions=0):
        return {
            "event_type": "run_summary", "run_id": run_id, "policy": policy,
            "task_id": task_id, "seed": seed,
            "scenario_hash": f"scenario-{task_id}-{seed}",
            "initial_state_hash": "initial-empty-state", "fault_enabled": seed % 2 == 0,
            "passed": passed, "aborted": aborted, "steps": steps,
            "steps_executed": executed, "intervention_count": interventions,
            "latency_seconds": 0.1,
        }

    def test_metric_calculations_use_matched_control_denominators(self):
        rows = []
        rows.extend([
            self.summary("a-no", "no_intervention", "task_01", 0, True),
            self.summary("a-naive", "naive_threshold", "task_01", 0, True),
            self.summary("a-proposed", "proposed", "task_01", 0, True, interventions=1),
            self.summary("b-no", "no_intervention", "task_01", 1, False),
            self.summary("b-naive", "naive_threshold", "task_01", 1, False, aborted=True, executed=0),
            self.summary("b-proposed", "proposed", "task_01", 1, True, interventions=1),
        ])
        metrics = compute_metrics(rows)
        self.assertEqual(metrics["no_intervention"]["task_success_rate"], 0.5)
        self.assertEqual(metrics["naive_threshold"]["average_steps"], 0.5)
        self.assertEqual(metrics["proposed"]["intervention_count"], 2)
        self.assertEqual(metrics["proposed"]["intervention_rate"], 1.0)
        self.assertEqual(metrics["proposed"]["disruption_count"], 1)
        self.assertEqual(metrics["proposed"]["disruption_denominator"], 1)
        self.assertEqual(metrics["proposed"]["recovery_count"], 1)
        self.assertEqual(metrics["proposed"]["recovery_denominator"], 1)

    def test_matched_evaluation_is_reproducible_and_writes_actual_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            model = root / "model.joblib"
            train_and_save(model, root / "training.jsonl", seeds=range(4))
            with patch("policy.train_policy.simulate_counterfactual_branch", side_effect=AssertionError("evaluation branched")) as branch, \
                    patch("policy.train_policy.train_and_save", side_effect=AssertionError("evaluation retrained")) as train:
                first = evaluate_suite(model, root / "first", tasks=TASKS[:2], seeds=range(2))
                second = evaluate_suite(model, root / "second", tasks=TASKS[:2], seeds=range(2))
            branch.assert_not_called()
            train.assert_not_called()
            self.assertEqual(first["scenario_count"], 4)
            self.assertEqual(first["run_count"], 12)
            first_summaries = load_run_summaries(first["trace"])
            second_summaries = load_run_summaries(second["trace"])
            validate_matched_runs(first_summaries)
            stable = lambda values: sorted(
                (x["task_id"], x["seed"], x["policy"], x["scenario_hash"], x["initial_state_hash"],
                 x["fault_enabled"], x["passed"], x["aborted"], x["steps_executed"], x["intervention_count"])
                for x in values
            )
            self.assertEqual(stable(first_summaries), stable(second_summaries))
            self.assertTrue(compare_repetitions(first["trace"], second["trace"]))
            self.assertEqual(first["metrics"]["no_intervention"]["successes"], 2)
            self.assertEqual(first["metrics"]["naive_threshold"]["successes"], 2)
            self.assertEqual(first["metrics"]["proposed"]["successes"], 2)
            for key in ("trace", "table", "csv", "plot"):
                self.assertTrue(first[key].is_file())
            self.assertIn("2/4 (50.0%)", first["table"].read_text(encoding="utf-8"))
            self.assertIn("<svg", first["plot"].read_text(encoding="utf-8"))

    def test_default_evaluation_uses_artifact_declared_heldout_tasks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            model = Path(directory) / "model.joblib"
            train_and_save(model, Path(directory) / "training.jsonl", seeds=range(4))
            result = evaluate_suite(
                model, Path(directory) / "evaluation", seeds=range(2)
            )
            self.assertEqual(result["task_ids"], ["task_07", "task_08"])
            self.assertEqual(result["scenario_count"], 4)

    def test_log_loader_rejects_missing_fields_invalid_json_and_aborted_success(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text('{"event_type":"run_summary","policy":"proposed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_run_summaries(path)
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_run_summaries(path)
            row = self.summary("x", "proposed", "task_01", 0, True, aborted=True)
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aborted run cannot pass"):
                load_run_summaries(path)

    def test_log_loader_rejects_unmatched_scenarios(self):
        rows = [
            self.summary("a", "no_intervention", "task_01", 0, True),
            self.summary("b", "naive_threshold", "task_01", 0, True),
            self.summary("c", "proposed", "task_01", 0, True),
        ]
        rows[-1]["initial_state_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "mismatched initial_state_hash"):
            validate_matched_runs(rows)


if __name__ == "__main__":
    unittest.main()
