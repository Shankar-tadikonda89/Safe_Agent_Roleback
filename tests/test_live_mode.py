import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.live_gemini import (
    LOCKED_MODEL_ID,
    GeminiJsonClient,
    LiveGeminiError,
    ProjectGeminiConfig,
    load_project_gemini_config,
    run_live,
)
from run_demo import TASK


class FakeGateway:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_json(self, role, prompt, seed):
        self.calls.append((role, prompt, seed))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, {"prompt_tokens": 10, "output_tokens": 4, "total_tokens": 14}


class FakeSdkModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class LiveGeminiTests(unittest.TestCase):
    def test_only_project_local_env_is_used_and_model_is_locked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            with patch("agents.live_gemini.PROJECT_ROOT", root), patch.dict(os.environ, {"GEMINI_API_KEY": "process-secret"}):
                with self.assertRaisesRegex(LiveGeminiError, "missing_api_key"):
                    load_project_gemini_config()
                (root / ".env").write_text(
                    "GEMINI_API_KEY=local-test-secret\nGEMINI_MODEL=gemini-3.8-flash\n", encoding="utf-8"
                )
                config = load_project_gemini_config()
                self.assertEqual(config.api_key, "local-test-secret")
                self.assertEqual(config.model, LOCKED_MODEL_ID)
                (root / ".env").write_text(
                    "GEMINI_API_KEY=local-test-secret\nGEMINI_MODEL=another-model\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(LiveGeminiError, "locked_model_mismatch"):
                    load_project_gemini_config()

    def test_sdk_call_uses_locked_model_seed_json_mode_and_token_usage(self):
        response = SimpleNamespace(
            text='{"failure_prob":0.25}',
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2, total_token_count=7),
        )
        models = FakeSdkModels(response=response)
        client = GeminiJsonClient(ProjectGeminiConfig("test-only-secret", LOCKED_MODEL_ID), SimpleNamespace(models=models))
        parsed, usage = client.generate_json("critic", "safe prompt", 17)
        request = models.calls[0]
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertEqual(request["contents"], "safe prompt")
        self.assertEqual(request["config"].seed, 17)
        self.assertEqual(request["config"].temperature, 0.0)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertEqual(parsed, {"failure_prob": 0.25})
        self.assertEqual(usage["total_tokens"], 7)

    def test_provider_errors_are_sanitized_and_not_retried(self):
        error = RuntimeError("request failed with key test-only-secret")
        error.code = 429
        models = FakeSdkModels(error=error)
        client = GeminiJsonClient(ProjectGeminiConfig("test-only-secret", LOCKED_MODEL_ID), SimpleNamespace(models=models))
        with self.assertRaises(LiveGeminiError) as raised:
            client.generate_json("planner", "prompt", 1)
        self.assertEqual(raised.exception.code, "rate_limit")
        self.assertEqual(raised.exception.provider_status, 429)
        self.assertNotIn("test-only-secret", str(raised.exception))
        self.assertEqual(len(models.calls), 1)

    def test_auth_model_and_server_errors_are_classified(self):
        expected = {401: "authentication_or_permission", 403: "authentication_or_permission",
                    404: "model_not_found_or_unavailable", 503: "provider_unavailable"}
        for status, code in expected.items():
            with self.subTest(status=status):
                error = RuntimeError("sanitized test exception")
                error.code = status
                models = FakeSdkModels(error=error)
                client = GeminiJsonClient(ProjectGeminiConfig("secret", LOCKED_MODEL_ID), SimpleNamespace(models=models))
                with self.assertRaises(LiveGeminiError) as raised:
                    client.generate_json("coder", "prompt", 9)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.provider_status, status)
                self.assertEqual(len(models.calls), 1)

    def test_auth_model_and_server_errors_are_classified(self):
        expected = {401: "authentication_or_permission", 403: "authentication_or_permission",
                    404: "model_not_found_or_unavailable", 503: "provider_unavailable"}
        for status, code in expected.items():
            with self.subTest(status=status):
                error = RuntimeError("sanitized test exception")
                error.code = status
                models = FakeSdkModels(error=error)
                client = GeminiJsonClient(ProjectGeminiConfig("secret", LOCKED_MODEL_ID), SimpleNamespace(models=models))
                with self.assertRaises(LiveGeminiError) as raised:
                    client.generate_json("coder", "prompt", 9)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.provider_status, status)
                self.assertEqual(len(models.calls), 1)

    def test_timeout_and_malformed_json_fail_cleanly(self):
        timeout_models = FakeSdkModels(error=TimeoutError("network timed out"))
        timeout_client = GeminiJsonClient(ProjectGeminiConfig("secret", LOCKED_MODEL_ID), SimpleNamespace(models=timeout_models))
        with self.assertRaises(LiveGeminiError) as timeout:
            timeout_client.generate_json("planner", "prompt", 1)
        self.assertEqual(timeout.exception.code, "timeout")

        malformed_models = FakeSdkModels(response=SimpleNamespace(text="not-json", usage_metadata=None))
        malformed_client = GeminiJsonClient(ProjectGeminiConfig("secret", LOCKED_MODEL_ID), SimpleNamespace(models=malformed_models))
        with self.assertRaises(LiveGeminiError) as malformed:
            malformed_client.generate_json("planner", "prompt", 1)
        self.assertEqual(malformed.exception.code, "malformed_response")

    def test_live_roles_respect_prompt_boundaries_and_oracle_is_authoritative(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / ".env").write_text("GEMINI_API_KEY=do-not-log-this\n", encoding="utf-8")
            fake = FakeGateway([
                {"steps": ["Compute the requested result and assign answer.", "Keep answer available."]},
                {"failure_prob": 0.1},
                {"code": "answer = 40\n"},
                {"failure_prob": 0.1},
                {"code": ""},
                {"verified": True, "reason": "model review says pass"},
            ])
            with patch("agents.live_gemini.PROJECT_ROOT", root):
                result = run_live(TASK, 42, "no_intervention", root / "live.jsonl", lambda config: fake)
            self.assertEqual([call[0] for call in fake.calls], ["planner", "critic", "coder", "critic", "coder", "verifier"])
            self.assertIn(TASK.prompt, fake.calls[0][1])
            coder_prompts = [prompt for role, prompt, _ in fake.calls if role == "coder"]
            self.assertTrue(all(TASK.prompt not in prompt for prompt in coder_prompts))
            self.assertTrue(all("fault_enabled" not in prompt for role, prompt, _ in fake.calls if role == "critic"))
            self.assertTrue(all(seed == 42 for _, _, seed in fake.calls))
            summary = result["summary"]
            self.assertEqual(summary["mode"], "live")
            self.assertEqual(summary["model_id"], LOCKED_MODEL_ID)
            self.assertEqual(summary["llm_calls"], 6)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["final_result"], 40)
            entries = [json.loads(line) for line in result["log_path"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(entries[-1]["status"], "failed")
            self.assertFalse(entries[-2]["verifier"]["model_agrees_with_oracle"])
            self.assertNotIn("do-not-log-this", result["log_path"].read_text(encoding="utf-8"))

    def test_missing_key_and_provider_failure_write_valid_failure_records(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            with patch("agents.live_gemini.PROJECT_ROOT", root):
                missing = run_live(TASK, 42, log_path=root / "missing.jsonl", gateway_factory=lambda _: self.fail("gateway must not initialize"))
                self.assertEqual(missing["error"]["code"], "missing_api_key")
                self.assertEqual(missing["summary"]["llm_calls"], 0)

                (root / ".env").write_text("GEMINI_API_KEY=secret-only\n", encoding="utf-8")
                gateway = FakeGateway([LiveGeminiError("rate_limit", "planner", 429)])
                failed = run_live(TASK, 42, log_path=root / "failed.jsonl", gateway_factory=lambda _: gateway)
            self.assertFalse(failed["summary"]["passed"])
            self.assertEqual(failed["summary"]["status"], "error")
            self.assertEqual(failed["summary"]["error"]["code"], "rate_limit")
            rows = [json.loads(line) for line in failed["log_path"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["event_type"], "run_summary")
            self.assertTrue(any(row["event_type"] == "live_error" for row in rows))
            self.assertNotIn("secret-only", failed["log_path"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
