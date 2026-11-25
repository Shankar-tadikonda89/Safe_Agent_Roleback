"""Locked-model Gemini roles and a separate, single-task live trajectory."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import dotenv_values

from run_demo import (
    NaiveThresholdPolicy,
    NoInterventionPolicy,
    Scenario,
    Verifier,
    estimate_failure_probability,
    extract_features,
    make_scenario,
)


LOCKED_MODEL_ID = "gemini-3.8-flash"
PROMPT_TEMPLATE_VERSION = "live-v1"
REQUEST_TIMEOUT_MS = 15_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_LOG = PROJECT_ROOT / "logs" / "live_runs.jsonl"


class LiveGeminiError(RuntimeError):
    """Sanitized provider/config error suitable for JSONL logging."""

    def __init__(self, code: str, stage: str, provider_status: int | None = None):
        super().__init__(f"{code} during {stage}")
        self.code = code
        self.stage = stage
        self.provider_status = provider_status


@dataclass(frozen=True)
class ProjectGeminiConfig:
    api_key: str
    model: str


def load_project_gemini_config() -> ProjectGeminiConfig:
    """Read only this project's .env; never fall back to process/home credentials."""
    env_file = PROJECT_ROOT / ".env"
    values = dotenv_values(env_file, interpolate=False) if env_file.is_file() else {}
    api_key = values.get("GEMINI_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise LiveGeminiError("missing_api_key", "configuration")
    model = values.get("GEMINI_MODEL") or LOCKED_MODEL_ID
    if model != LOCKED_MODEL_ID:
        raise LiveGeminiError("locked_model_mismatch", "configuration")
    return ProjectGeminiConfig(api_key=api_key.strip(), model=LOCKED_MODEL_ID)


class GeminiJsonClient:
    """Small JSON-only wrapper; one request per role call and no automatic retries."""

    def __init__(self, config: ProjectGeminiConfig, sdk_client: Any | None = None):
        self.config = config
        if sdk_client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LiveGeminiError("sdk_unavailable", "client_initialization") from exc
            sdk_client = genai.Client(
                api_key=config.api_key,
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._types = types
        else:
            from google.genai import types

            self._types = types
        self.client = sdk_client

    def generate_json(self, stage: str, prompt: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            seed=int(seed),
            max_output_tokens=512,
        )
        try:
            response = self.client.models.generate_content(
                model=LOCKED_MODEL_ID,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            status = getattr(exc, "code", None)
            if not isinstance(status, int):
                status = getattr(exc, "status_code", None)
            name = type(exc).__name__.lower()
            if "timeout" in name or isinstance(exc, TimeoutError):
                code = "timeout"
            elif status == 429:
                code = "rate_limit"
            elif status in (401, 403):
                code = "authentication_or_permission"
            elif status == 404:
                code = "model_not_found_or_unavailable"
            elif isinstance(status, int) and status >= 500:
                code = "provider_unavailable"
            else:
                code = "provider_error"
            raise LiveGeminiError(code, stage, status if isinstance(status, int) else None) from None

        try:
            raw_text = response.text
            parsed = json.loads(raw_text)
        except Exception:
            raise LiveGeminiError("malformed_response", stage) from None
        if not isinstance(parsed, dict):
            raise LiveGeminiError("malformed_response", stage)
        usage = getattr(response, "usage_metadata", None)
        telemetry = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return parsed, telemetry


def _require(condition: bool, stage: str) -> None:
    if not condition:
        raise LiveGeminiError("malformed_response", stage)


class LivePlanner:
    def __init__(self, generate_json: Callable):
        self.generate_json = generate_json

    def plan(self, task: Any, seed: int) -> tuple[list[str], dict[str, Any]]:
        prompt = (
            "Role: Planner. You alone may see the full task description. Decompose it into 2 to 4 "
            "short, ordered, executable step instructions. Return JSON exactly as "
            '{"steps":["instruction", ...]}. Do not solve the task in hidden metadata.\n'
            f"Full task description:\n{task.prompt}"
        )
        response, telemetry = self.generate_json("planner", prompt, seed)
        steps = response.get("steps")
        _require(isinstance(steps, list) and 2 <= len(steps) <= 4, "planner")
        _require(all(isinstance(step, str) and step.strip() for step in steps), "planner")
        return [step.strip() for step in steps], telemetry

    def replan(self, task: Any, current_instruction: str, guidance: str, seed: int) -> tuple[str, dict[str, Any]]:
        prompt = (
            "Role: Planner. Replan only the current step from the current trajectory state. "
            "Use the task and corrective guidance; do not ask to restore or replay an earlier state. "
            'Return JSON exactly as {"instruction":"corrected current step"}.\n'
            f"Task description (Planner only):\n{task.prompt}\n"
            f"Current step instruction:\n{current_instruction}\n"
            f"Corrective guidance:\n{guidance}"
        )
        response, telemetry = self.generate_json("planner.replan", prompt, seed)
        instruction = response.get("instruction")
        _require(isinstance(instruction, str) and instruction.strip(), "planner.replan")
        return instruction.strip(), telemetry


class LiveCoder:
    def __init__(self, generate_json: Callable):
        self.generate_json = generate_json

    def implement(self, current_instruction: str, accumulated_code: str, seed: int) -> tuple[str, dict[str, Any]]:
        prompt = (
            "Role: Coder. Follow only the current step instruction and accumulated code shown below. "
            "Do not reconstruct or request the original task prompt. Return JSON exactly as "
            '{"code":"new code text to append"}. Generate inert Python text only; never provide shell commands.\n'
            f"Accumulated code:\n{accumulated_code}\n"
            f"Current step instruction:\n{current_instruction}"
        )
        response, telemetry = self.generate_json("coder", prompt, seed)
        code = response.get("code")
        _require(isinstance(code, str) and len(code) <= 8000, "coder")
        return code, telemetry


class LiveCritic:
    def __init__(self, generate_json: Callable):
        self.generate_json = generate_json

    def estimate(self, current_instruction: str, code_prefix: str, position: float, seed: int) -> tuple[float, dict[str, Any]]:
        prompt = (
            "Role: trajectory-prefix risk critic. Estimate the probability that continuing from this "
            "visible prefix would fail its deterministic task oracle. Return JSON exactly as "
            '{"failure_prob":0.0}. Use a finite number from 0 through 1. Do not infer or request '
            "injected-fault flags, expected answers, future steps, labels, or verifier outcomes.\n"
            f"Current step instruction:\n{current_instruction}\n"
            f"Accumulated code prefix:\n{code_prefix}\n"
            f"Trajectory position (0 to 1): {position:.6f}"
        )
        response, telemetry = self.generate_json("critic", prompt, seed)
        value = response.get("failure_prob")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise LiveGeminiError("malformed_response", "critic")
        return float(value), telemetry


class LiveVerifier:
    def __init__(self, generate_json: Callable):
        self.generate_json = generate_json

    def review(self, code: str, expected: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = (
            "Role: Verifier. Independently compare the final code's integer answer with the supplied "
            "task-oracle output. Return JSON exactly as {\"verified\":true,\"reason\":\"short reason\"}.\n"
            f"Oracle expected integer: {expected}\nFinal code:\n{code}"
        )
        response, telemetry = self.generate_json("verifier", prompt, seed)
        verified = response.get("verified")
        reason = response.get("reason")
        _require(isinstance(verified, bool) and isinstance(reason, str), "verifier")
        return {"verified": verified, "reason": reason[:500]}, telemetry


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def run_live(
    task: Any,
    seed: int,
    policy_name: str = "no_intervention",
    log_path: str | Path = DEFAULT_LIVE_LOG,
    gateway_factory: Callable | None = None,
    model_path: str | Path = Path("models/proposed_policy.joblib"),
) -> dict[str, Any]:
    """Run a single live task. Provider failures are logged as failures, never passes."""
    log_path = Path(log_path)
    scenario: Scenario = make_scenario(task, seed)
    initial_state = {"code": "", "completed_steps": []}
    import hashlib

    initial_state_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
    run_id = str(uuid.uuid4())
    events: list[dict[str, Any]] = []
    call_records: list[dict[str, Any]] = []
    attempted_calls = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    code = ""
    status = "error"
    passed = False
    aborted = False
    final_result = None
    inference_calls = 0
    started = time.perf_counter()
    stage = "configuration"
    try:
        if policy_name not in {"no_intervention", "naive_threshold", "proposed"}:
            raise LiveGeminiError("unknown_policy", "configuration")
        proposed_policy = None
        if policy_name == "proposed":
            from policy.proposed import ProposedPolicy

            proposed_policy = ProposedPolicy.load(model_path)
        config = load_project_gemini_config()
        gateway = gateway_factory(config) if gateway_factory else GeminiJsonClient(config)

        def generate_json(role: str, prompt: str, request_seed: int):
            nonlocal attempted_calls, total_prompt_tokens, total_output_tokens
            attempted_calls += 1
            try:
                parsed, usage = gateway.generate_json(role, prompt, request_seed)
            except LiveGeminiError as exc:
                call_records.append({"role": role, "success": False, "error_code": exc.code,
                                     "provider_status": exc.provider_status})
                raise
            call_records.append({"role": role, "success": True, "prompt_tokens": usage.get("prompt_tokens"),
                                 "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens")})
            if isinstance(usage.get("prompt_tokens"), int):
                total_prompt_tokens += usage["prompt_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                total_output_tokens += usage["output_tokens"]
            return parsed, usage

        planner = LivePlanner(generate_json)
        coder = LiveCoder(generate_json)
        critic = LiveCritic(generate_json)
        verifier = LiveVerifier(generate_json)
        stage = "planner"
        instructions, _ = planner.plan(task, seed)
        prior_interventions = 0
        for step_index, canonical_instruction in enumerate(instructions):
            stage = "critic"
            current_instruction = canonical_instruction
            if scenario.fault_enabled and step_index == scenario.fault_step:
                current_instruction = f"Return the integer result {scenario.wrong_answer}."
            position = step_index / max(len(instructions) - 1, 1)
            risk, _ = critic.estimate(current_instruction, code, position, seed)
            features = extract_features(risk, "code_assignment", step_index, len(instructions), prior_interventions)
            if policy_name == "no_intervention":
                decision = NoInterventionPolicy().decide(risk, features)
                decision_details = None
            elif policy_name == "naive_threshold":
                decision = NaiveThresholdPolicy().decide(risk, features)
                decision_details = None
            else:
                decision_details = proposed_policy.decide(features)
                decision = decision_details["decision"]
                inference_calls += 1
            code_before = code
            guidance = None
            if decision == "abort":
                aborted = True
                events.append({
                    "event_type": "step", "run_id": run_id, "mode": "live", "policy": policy_name,
                    "task_id": task.task_id, "seed": seed, "scenario_hash": scenario.scenario_hash,
                    "initial_state": initial_state, "initial_state_hash": initial_state_hash,
                    "fault_enabled": scenario.fault_enabled, "fault_type": scenario.fault_type,
                    "step_index": step_index, "instruction": current_instruction,
                    "code_before": code_before, "code": code, "failure_prob": risk,
                    "features": features, "decision": "abort", "intervention": False,
                    "corrective_guidance": None, "planner_replanned": False,
                    "policy_decision_details": decision_details, "branch_type": "live",
                    "step_executed": False, "verifier": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                break
            if decision == "intervene":
                guidance = "Correct the current step if it is inconsistent with the task; preserve prior code and continue from this state."
                stage = "planner.replan"
                current_instruction, _ = planner.replan(task, current_instruction, guidance, seed)
                current_instruction = f"{current_instruction}\nGuidance: {guidance}"
                prior_interventions += 1
            if decision not in {"continue", "intervene"}:
                raise LiveGeminiError("invalid_policy_decision", stage)
            stage = "coder"
            code_append, _ = coder.implement(current_instruction, code, seed)
            code = code + ("\n" if code and code_append and not code.endswith("\n") else "") + code_append
            # The generated text is inert and is never executed as a subprocess or shell command.
            events.append({
                "event_type": "step", "run_id": run_id, "mode": "live", "policy": policy_name,
                "task_id": task.task_id, "seed": seed, "scenario_hash": scenario.scenario_hash,
                "initial_state": initial_state, "initial_state_hash": initial_state_hash,
                "fault_enabled": scenario.fault_enabled, "fault_type": scenario.fault_type,
                "fault_step": scenario.fault_step, "step_index": step_index,
                "instruction": current_instruction, "faulted_instruction": (
                    f"Return the integer result {scenario.wrong_answer}."
                    if scenario.fault_enabled and step_index == scenario.fault_step else canonical_instruction
                ),
                "code_before": code_before, "code": code, "failure_prob": risk,
                "critic_source": "gemini_live", "features": features, "decision": decision,
                "intervention": decision == "intervene", "corrective_guidance": guidance,
                "planner_replanned": decision == "intervene", "policy_decision_details": decision_details,
                "branch_type": "live", "step_executed": True, "verifier": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        local_passed = False
        local_evidence = None
        model_review = None
        if not aborted:
            stage = "verifier"
            model_review, _ = verifier.review(code, task.expected, seed)
            local_passed, local_evidence = Verifier().verify(code, task)
            passed = local_passed
            final_result = local_evidence["actual"]
            status = "complete" if local_passed else "failed"
            if events:
                events[-1]["verifier"] = {**local_evidence, "model_review": model_review,
                                             "model_agrees_with_oracle": model_review["verified"] == local_passed}
        else:
            status = "aborted"
        status = status or "failed"
    except LiveGeminiError as exc:
        failure = {"code": exc.code, "stage": exc.stage, "provider_status": exc.provider_status}
    except Exception as exc:
        # Never serialize provider exception text: SDK exceptions can contain sensitive request details.
        failure = {"code": "internal_live_error", "stage": stage, "provider_status": None,
                   "exception_type": type(exc).__name__}
    else:
        failure = None

    if failure:
        status = "error"
        passed = False
        final_result = None
        events.append({
            "event_type": "live_error", "run_id": run_id, "mode": "live", "policy": policy_name,
            "task_id": task.task_id, "seed": seed, "scenario_hash": scenario.scenario_hash,
            "initial_state_hash": initial_state_hash, "branch_type": "live", "error": failure,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    summary = {
        "event_type": "run_summary", "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live", "status": status, "policy": policy_name, "task_id": task.task_id,
        "seed": seed, "scenario_hash": scenario.scenario_hash,
        "initial_state": initial_state, "initial_state_hash": initial_state_hash,
        "fault_enabled": scenario.fault_enabled, "passed": bool(passed), "aborted": aborted,
        "final_result": final_result, "steps": len([e for e in events if e["event_type"] == "step"]),
        "steps_executed": sum(bool(e.get("step_executed")) for e in events if e["event_type"] == "step"),
        "intervention_count": sum(bool(e.get("intervention")) for e in events if e["event_type"] == "step"),
        "latency_seconds": time.perf_counter() - started,
        "llm_calls": attempted_calls, "policy_inference_calls": inference_calls,
        "prompt_tokens": total_prompt_tokens if attempted_calls else 0,
        "output_tokens": total_output_tokens if attempted_calls else 0,
        "api_calls": call_records,
        "model_id": LOCKED_MODEL_ID, "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "temperature": 0.0, "model_seed": seed, "critic_source": "gemini_live",
    }
    if failure:
        summary["error"] = failure
    _append_jsonl(log_path, [*events, summary])
    return {"events": events, "summary": summary, "error": failure, "log_path": log_path}
