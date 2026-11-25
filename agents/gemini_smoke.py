"""Single-request, sanitized Gemini smoke diagnostic.

The CLI is intentionally not run by the offline test suite. Tests inject local
configuration and fake SDK clients so they cannot contact a provider.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

from agents.live_gemini import (
    LOCKED_MODEL_ID,
    GeminiJsonClient,
    LiveGeminiError,
    ProjectGeminiConfig,
    load_project_gemini_config,
)


SMOKE_PROMPT = 'Return exactly {"ok": true} as JSON.'


@dataclass(frozen=True)
class SmokeResult:
    model_id: str
    api_key_configured: bool | None
    category: str | None

    @property
    def passed(self) -> bool:
        return self.category is None

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def _failure_category(error: BaseException) -> str:
    """Map known sanitized errors to fixed, non-sensitive categories."""
    code = error.code if isinstance(error, LiveGeminiError) else None
    if code in {"authentication_or_permission"}:
        return "authentication_error"
    if code == "timeout":
        return "timeout"
    if code == "rate_limit":
        return "rate_limit"
    if code == "malformed_response":
        return "malformed_response"
    return "provider_error"


def run_smoke(
    config_loader: Callable[[], ProjectGeminiConfig] = load_project_gemini_config,
    client_factory: Callable[[ProjectGeminiConfig], Any] = GeminiJsonClient,
) -> SmokeResult:
    """Load local settings and perform at most one JSON call; never log details."""
    model_id = "unavailable"
    key_configured: bool | None = None
    try:
        config = config_loader()
        model_id = config.model
        key_configured = bool(config.api_key)
        if model_id != LOCKED_MODEL_ID or not key_configured:
            return SmokeResult(model_id, key_configured, "initialization_error")
        client = client_factory(config)
    except BaseException:
        return SmokeResult(model_id, key_configured, "initialization_error")

    try:
        response, _ = client.generate_json("smoke", SMOKE_PROMPT, 42)
    except BaseException as error:
        return SmokeResult(model_id, key_configured, _failure_category(error))

    if not isinstance(response, dict) or response.get("ok") is not True:
        return SmokeResult(model_id, key_configured, "unexpected_response")
    return SmokeResult(model_id, key_configured, None)


def main() -> int:
    result = run_smoke()
    print(f"model_id={result.model_id}")
    print(f"api_key_configured={result.api_key_configured}")
    print("smoke=PASS" if result.passed else f"smoke=FAIL:{result.category}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
