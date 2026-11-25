"""Versioned artifact loading and single-pass cost-sensitive inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from run_demo import FEATURE_NAMES


ARTIFACT_VERSION = 2
RISK_SIGNAL_ID = "neutral_uninformative_prior_0.5"
DEFAULT_ARTIFACT = Path("models/proposed_policy.joblib")


class ProposedPolicy:
    """Predict helpful-intervention probability once, then minimize expected cost."""

    name = "proposed"

    def __init__(self, artifact: dict[str, Any]):
        self._validate_artifact(artifact)
        self.artifact = artifact
        self.model: Pipeline = artifact["model"]

    @staticmethod
    def _validate_artifact(artifact: Any) -> None:
        if not isinstance(artifact, dict):
            raise ValueError("policy artifact must be a metadata dictionary")
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported policy artifact version")
        metadata = artifact.get("training_metadata")
        if not isinstance(metadata, dict) or metadata.get("risk_signal") != RISK_SIGNAL_ID:
            raise ValueError("policy artifact does not declare the current non-leaking risk signal")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("policy artifact feature order does not match the required feature schema")
        if artifact.get("target") != "intervene_helped":
            raise ValueError("policy artifact target is incompatible")
        model = artifact.get("model")
        if not isinstance(model, Pipeline) or not hasattr(model, "predict_proba"):
            raise ValueError("policy artifact does not contain a fitted probability classifier")
        classes = set(model.classes_.tolist())
        if classes != {0, 1}:
            raise ValueError("policy classifier must contain both target classes 0 and 1")
        costs = artifact.get("cost_matrix")
        if costs != {"continue": {"not_helpful": 0.0, "helpful": 3.0},
                     "intervene": {"not_helpful": 2.0, "helpful": 1.0}}:
            raise ValueError("policy artifact cost matrix is missing or incompatible")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "ProposedPolicy":
        try:
            artifact = joblib.load(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"trained proposed policy not found at {path}; run python -m policy.train_policy first"
            ) from exc
        return cls(artifact)

    def decide(self, features: dict[str, float]) -> dict[str, float | str]:
        if tuple(features) != FEATURE_NAMES:
            raise ValueError(f"features must be supplied in this exact order: {FEATURE_NAMES}")
        row = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
        if row.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(row).all():
            raise ValueError("features must be one finite four-value row")
        if ((row < 0.0) | (row > 1.0)).any():
            raise ValueError("features must be in [0, 1]")

        # The only model call in proposed inference. No branch/rollback helpers are available here.
        probabilities = self.model.predict_proba(row)
        class_index = list(self.model.classes_).index(1)
        helpful_probability = float(probabilities[0, class_index])
        costs = self.artifact["cost_matrix"]
        continue_cost = (
            helpful_probability * costs["continue"]["helpful"]
            + (1.0 - helpful_probability) * costs["continue"]["not_helpful"]
        )
        intervene_cost = (
            helpful_probability * costs["intervene"]["helpful"]
            + (1.0 - helpful_probability) * costs["intervene"]["not_helpful"]
        )
        decision = "intervene" if intervene_cost < continue_cost else "continue"
        if not math.isfinite(helpful_probability) or not 0.0 <= helpful_probability <= 1.0:
            raise ValueError("classifier returned an invalid probability")
        return {
            "decision": decision,
            "helpful_intervention_probability": helpful_probability,
            "expected_continue_cost": float(continue_cost),
            "expected_intervene_cost": float(intervene_cost),
        }
