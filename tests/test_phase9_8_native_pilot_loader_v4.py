from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.phase9_8_adapter import MODEL_PATH, _canonical_bytes
from experiments.phase9_8_native_pilot_loader_v4 import (
    NATIVE_PILOT_LOCK_V4_PATH, NativePilotV4ArtifactError,
    WRAPPER_KEYS, load_native_pilot_adapter_v4,
)


def _fitted_pipeline_fixture() -> Pipeline:
    """Construct shape-correct fitted state without fitting/training an estimator."""
    scaler = StandardScaler()
    scaler.n_features_in_ = 4
    scaler.mean_ = np.zeros(4)
    scaler.var_ = np.ones(4)
    scaler.scale_ = np.ones(4)
    scaler.n_samples_seen_ = 32

    classifier = LogisticRegression(C=1.0)
    classifier.n_features_in_ = 4
    classifier.classes_ = np.asarray([0, 1, 2, 3])
    classifier.coef_ = np.zeros((4, 4))
    classifier.intercept_ = np.zeros(4)
    classifier.n_iter_ = np.asarray([1])

    pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
    return pipeline


class NativePilotV4LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path.cwd()
        cls.draft_lock = json.loads(
            (cls.project_root / NATIVE_PILOT_LOCK_V4_PATH).read_text(encoding="utf-8")
        )

    def _temporary_approved_lock(self, folder: Path) -> Path:
        lock = copy.deepcopy(self.draft_lock)
        lock["approval_status"] = "APPROVED"  # temporary test-only copy
        lock.pop("lock_canonical_sha256", None)
        lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
        path = folder / "test_only_approved_v4_lock.json"
        path.write_text(
            json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        return path

    def _load_wrapper(self, wrapper: dict[str, object]):
        with tempfile.TemporaryDirectory(dir=self.project_root) as temp:
            lock_path = self._temporary_approved_lock(Path(temp))
            calls: list[Path] = []
            adapter = load_native_pilot_adapter_v4(
                root=self.project_root,
                lock_path=lock_path,
                approved_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                model_loader=lambda path: calls.append(path) or wrapper,
            )
            self.assertEqual(calls, [self.project_root / MODEL_PATH])
            return adapter

    def _wrapper(self) -> dict[str, object]:
        metadata = json.loads(
            (self.project_root / "models/phase9_8_final/model_metadata.json").read_text(encoding="utf-8")
        )
        return {
            "artifact_version": metadata["artifact_version"],
            "variant_name": "full_features",
            "pipeline": _fitted_pipeline_fixture(),
            "feature_names": list(metadata["feature_names"]),
            "class_names": list(metadata["class_names"]),
            "cost_matrix": copy.deepcopy(metadata["cost_matrix"]),
            "selected_C": metadata["selected_C"],
            "classifier_classes": list(metadata["classifier_classes"]),
            "objective": (
                "multinomial LogisticRegression maximum likelihood; frozen cost matrix is applied "
                "by expected-cost rule at inference"
            ),
        }

    def test_real_draft_lock_rejects_before_injected_artifact_loader(self):
        path = self.project_root / NATIVE_PILOT_LOCK_V4_PATH
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        calls: list[Path] = []
        with self.assertRaises(NativePilotV4ArtifactError):
            load_native_pilot_adapter_v4(
                root=self.project_root, lock_path=path, approved_lock_sha256=raw_hash,
                model_loader=lambda model_path: calls.append(model_path),
            )
        self.assertEqual(calls, [])

    def test_valid_training_wrapper_extracts_only_fitted_pipeline(self):
        wrapper = self._wrapper()
        adapter = self._load_wrapper(wrapper)
        self.assertIsInstance(adapter._model, Pipeline)
        self.assertIs(adapter._model, wrapper["pipeline"])
        self.assertEqual(set(wrapper), WRAPPER_KEYS)

    def test_invalid_wrapper_fields_fail_closed(self):
        mutations = {
            "version": lambda row: row.__setitem__("artifact_version", "wrong-version"),
            "feature_order": lambda row: row.__setitem__("feature_names", list(reversed(row["feature_names"]))),
            "class_order": lambda row: row.__setitem__("class_names", list(reversed(row["class_names"]))),
            "cost_matrix": lambda row: row["cost_matrix"].__setitem__("corrupt", True),
            "selected_C": lambda row: row.__setitem__("selected_C", 10.0),
            "declared_classifier_order": lambda row: row.__setitem__(
                "classifier_classes", list(reversed(row["classifier_classes"])),
            ),
            "missing_pipeline": lambda row: row.pop("pipeline"),
            "variant": lambda row: row.__setitem__("variant_name", "ablation"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                wrapper = self._wrapper()
                mutate(wrapper)
                with self.assertRaises(NativePilotV4ArtifactError):
                    self._load_wrapper(wrapper)

    def test_bad_fitted_class_order_and_missing_fitted_state_fail_closed(self):
        wrapper = self._wrapper()
        wrapper["pipeline"].named_steps["classifier"].classes_ = np.asarray([0, 1, 3, 2])
        with self.assertRaises(NativePilotV4ArtifactError):
            self._load_wrapper(wrapper)

        wrapper = self._wrapper()
        del wrapper["pipeline"].named_steps["classifier"].coef_
        with self.assertRaises(NativePilotV4ArtifactError):
            self._load_wrapper(wrapper)


if __name__ == "__main__":
    unittest.main()
