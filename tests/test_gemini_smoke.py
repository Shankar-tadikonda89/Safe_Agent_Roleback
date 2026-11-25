import json
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from agents.gemini_smoke import SmokeResult, main, run_smoke
from agents.live_gemini import (
    LOCKED_MODEL_ID,
    GeminiJsonClient,
    LiveGeminiError,
    ProjectGeminiConfig,
)


class FakeModels:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text, usage_metadata=None)


class FakeGateway:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_json(self, stage, prompt, seed):
        self.calls.append((stage, prompt, seed))
        if self.error is not None:
            raise self.error
        return self.response, {}


class GeminiSmokeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = ProjectGeminiConfig("offline-test-secret", LOCKED_MODEL_ID)

    def invoke(self, gateway):
        return run_smoke(lambda: self.config, lambda _config: gateway)

    def test_configuration_and_client_initialization_fail_safely(self):
        missing = run_smoke(
            lambda: (_ for _ in ()).throw(LiveGeminiError("missing_api_key", "configuration")),
            lambda _config: self.fail("client factory must not run"),
        )
        self.assertEqual(missing.category, "initialization_error")
        self.assertEqual(missing.exit_code, 1)

        failed_init = run_smoke(
            lambda: self.config,
            lambda _config: (_ for _ in ()).throw(RuntimeError("sensitive initialization details")),
        )
        self.assertEqual(failed_init.category, "initialization_error")
        self.assertNotIn("sensitive", repr(failed_init))

    def test_model_mismatch_is_initialization_failure_without_request(self):
        config = ProjectGeminiConfig("offline-test-secret", "different-model")
        gateway_calls = []
        result = run_smoke(lambda: config, lambda _config: gateway_calls.append(True))
        self.assertEqual(result.category, "initialization_error")
        self.assertEqual(gateway_calls, [])
        self.assertEqual(result.exit_code, 1)

    def test_authentication_timeout_and_rate_limit_have_fixed_categories(self):
        cases = [
            (401, "authentication_error"),
            (403, "authentication_error"),
            (TimeoutError("sensitive timeout details"), "timeout"),
            (429, "rate_limit"),
        ]
        for failure, expected in cases:
            with self.subTest(expected=expected, failure=type(failure).__name__):
                error = failure if isinstance(failure, BaseException) else RuntimeError("provider failure")
                if isinstance(failure, int):
                    error.code = failure
                models = FakeModels(error=error)
                client = GeminiJsonClient(self.config, SimpleNamespace(models=models))
                result = self.invoke(client)
                self.assertEqual(result.category, expected)
                self.assertEqual(len(models.calls), 1)
                self.assertNotIn("sensitive", repr(result))

    def test_malformed_and_non_dictionary_json_are_malformed_response(self):
        for body in ("not-json", '["not", "an", "object"]'):
            with self.subTest(body=body):
                models = FakeModels(text=body)
                client = GeminiJsonClient(self.config, SimpleNamespace(models=models))
                result = self.invoke(client)
                self.assertEqual(result.category, "malformed_response")
                self.assertEqual(len(models.calls), 1)

    def test_valid_json_with_true_ok_passes_after_exactly_one_fake_call(self):
        models = FakeModels(text=json.dumps({"ok": True}))
        client = GeminiJsonClient(self.config, SimpleNamespace(models=models))
        result = self.invoke(client)
        self.assertTrue(result.passed)
        self.assertEqual(result.category, None)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(models.calls[0]["model"], LOCKED_MODEL_ID)

    def test_missing_or_false_ok_is_unexpected_response(self):
        for body in ({"other": True}, {"ok": False}, {"ok": 1}):
            with self.subTest(body=body):
                gateway = FakeGateway(response=body)
                result = self.invoke(gateway)
                self.assertEqual(result.category, "unexpected_response")
                self.assertEqual(result.exit_code, 1)
                self.assertEqual(len(gateway.calls), 1)

    def test_unexpected_request_exception_is_sanitized(self):
        gateway = FakeGateway(error=RuntimeError("key=offline-test-secret headers=private"))
        result = self.invoke(gateway)
        self.assertEqual(result.category, "provider_error")
        self.assertNotIn("offline-test-secret", repr(result))
        self.assertNotIn("private", repr(result))
        self.assertEqual(len(gateway.calls), 1)

    def test_cli_represents_failure_with_nonzero_exit_and_generic_output(self):
        output = io.StringIO()
        failure = SmokeResult(LOCKED_MODEL_ID, True, "rate_limit")
        with patch("agents.gemini_smoke.run_smoke", return_value=failure):
            with redirect_stdout(output):
                exit_code = main()
        self.assertEqual(exit_code, 1)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["model_id=gemini-3.8-flash", "api_key_configured=True", "smoke=FAIL:rate_limit"],
        )


if __name__ == "__main__":
    unittest.main()
