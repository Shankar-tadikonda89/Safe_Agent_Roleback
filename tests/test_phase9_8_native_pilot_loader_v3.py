from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase9_8_adapter import MODEL_PATH, _canonical_bytes
from experiments.phase9_8_native_pilot_loader_v3 import (
    NATIVE_PILOT_LOCK_V3_PATH, NativePilotV3ArtifactError,
    load_native_pilot_adapter_v3,
)


class _FakeModel:
    def __init__(self):
        scaler_type = type("StandardScaler", (), {})
        classifier_type = type("LogisticRegression", (), {})
        scaler = scaler_type()
        scaler.n_features_in_ = 4
        classifier = classifier_type()
        classifier.classes_ = [0, 1, 2, 3]
        classifier.C = 1.0
        self.named_steps = {"scaler": scaler, "classifier": classifier}
        self.n_features_in_ = 4

    def predict_proba(self, _matrix):
        return [[0.25, 0.25, 0.25, 0.25]]


class NativePilotV3LoaderTests(unittest.TestCase):
    def _fixture(self, folder: Path) -> Path:
        lock = json.loads(Path(NATIVE_PILOT_LOCK_V3_PATH).read_text(encoding="utf-8"))
        lock["approval_status"] = "APPROVED"  # temporary unit-test copy only
        lock.pop("lock_canonical_sha256", None)
        lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
        path = folder / "test_only_approved_lock.json"
        path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def test_actual_draft_lock_fails_before_fake_deserializer(self):
        path = Path(NATIVE_PILOT_LOCK_V3_PATH)
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        calls = []
        with self.assertRaises(NativePilotV3ArtifactError):
            load_native_pilot_adapter_v3(
                root=Path.cwd(), lock_path=path, approved_lock_sha256=raw,
                model_loader=lambda item: calls.append(item),
            )
        self.assertEqual(calls, [])

    def test_temporary_approved_copy_validates_before_fake_deserializer(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            path = self._fixture(Path(temp))
            raw = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = []
            adapter = load_native_pilot_adapter_v3(
                root=Path.cwd(), lock_path=path, approved_lock_sha256=raw,
                model_loader=lambda item: calls.append(item) or _FakeModel(),
            )
            self.assertEqual(calls, [Path.cwd() / MODEL_PATH])
            self.assertIsInstance(adapter._model, _FakeModel)

    def test_supersession_binding_mismatch_fails_before_fake_deserializer(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            path = self._fixture(Path(temp))
            lock = json.loads(path.read_text(encoding="utf-8"))
            lock["bound_file_sha256"]["superseding_preregistration_v3"] = "0" * 64
            lock.pop("lock_canonical_sha256", None)
            lock["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
            path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            raw = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = []
            with self.assertRaises(NativePilotV3ArtifactError):
                load_native_pilot_adapter_v3(
                    root=Path.cwd(), lock_path=path, approved_lock_sha256=raw,
                    model_loader=lambda item: calls.append(item),
                )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
