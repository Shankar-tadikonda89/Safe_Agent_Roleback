from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.phase9_8_native_pilot_loader_v2 import (
    NATIVE_PILOT_LOCK_V2_PATH, NativePilotArtifactError,
    load_native_pilot_adapter_v2,
)
from experiments.phase9_8_adapter import MODEL_PATH


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


class NativePilotV2LockGateTests(unittest.TestCase):
    def _approved_test_fixture(self, directory: Path) -> Path:
        """Create only a temporary test lock; the project lock stays draft."""
        project_lock = json.loads(Path(NATIVE_PILOT_LOCK_V2_PATH).read_text(encoding="utf-8"))
        project_lock["approval_status"] = "APPROVED"
        project_lock.pop("lock_canonical_sha256", None)
        import hashlib as _hashlib
        from experiments.phase9_8_adapter import _canonical_bytes
        project_lock["lock_canonical_sha256"] = _hashlib.sha256(_canonical_bytes(project_lock)).hexdigest()
        fixture = directory / "test-only-approved-lock.json"
        fixture.write_text(json.dumps(project_lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return fixture

    def test_valid_test_fixture_reaches_only_injected_fake_loader(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            lock = self._approved_test_fixture(Path(tmp))
            digest = hashlib.sha256(lock.read_bytes()).hexdigest()
            calls = []
            adapter = load_native_pilot_adapter_v2(
                root=Path.cwd(), lock_path=lock, approved_lock_sha256=digest,
                model_loader=lambda model_path: calls.append(model_path) or _FakeModel(),
            )
            self.assertEqual(calls, [Path.cwd() / MODEL_PATH])
            self.assertIsInstance(adapter._model, _FakeModel)

    def test_draft_lock_rejects_before_injected_deserializer(self):
        path = Path(NATIVE_PILOT_LOCK_V2_PATH)
        self.assertTrue(path.is_file())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        calls = []
        with self.assertRaises(NativePilotArtifactError):
            load_native_pilot_adapter_v2(
                root=Path.cwd(), lock_path=path,
                approved_lock_sha256=digest,
                model_loader=lambda _path: calls.append(_path),
            )
        self.assertEqual(calls, [])

    def test_missing_or_mismatched_external_digest_rejects_before_validation(self):
        path = Path(NATIVE_PILOT_LOCK_V2_PATH)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch("experiments.phase9_8_native_pilot_loader_v2._validate_lock") as validate:
            with self.assertRaises(NativePilotArtifactError):
                load_native_pilot_adapter_v2(root=Path.cwd(), lock_path=path,
                                             approved_lock_sha256=None,
                                             model_loader=lambda _path: object())
            with self.assertRaises(NativePilotArtifactError):
                load_native_pilot_adapter_v2(root=Path.cwd(), lock_path=path,
                                             approved_lock_sha256="0" * 64,
                                             model_loader=lambda _path: object())
        validate.assert_not_called()

    def test_policy_hash_partition_and_metadata_contract_mismatches_reject_before_loader(self):
        mutations = (
            lambda doc: doc.update(comparative_policy_set=["no_intervention"]),
            lambda doc: doc["bound_file_sha256"].update(pilot_runner="0" * 64),
            lambda doc: doc["pilot_scope"].update(scenario_count=2559),
            lambda doc: doc.update(frozen_feature_order=["reversibility_proxy"]),
            lambda doc: doc["inherited_frozen_raw_hashes"].update(model="0" * 64),
        )
        from experiments.phase9_8_adapter import _canonical_bytes
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                lock = self._approved_test_fixture(Path(tmp))
                document = json.loads(lock.read_text(encoding="utf-8"))
                document.pop("lock_canonical_sha256", None)
                mutate(document)
                document["lock_canonical_sha256"] = hashlib.sha256(_canonical_bytes(document)).hexdigest()
                lock.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                digest = hashlib.sha256(lock.read_bytes()).hexdigest()
                calls = []
                with self.assertRaises(NativePilotArtifactError):
                    load_native_pilot_adapter_v2(
                        root=Path.cwd(), lock_path=lock, approved_lock_sha256=digest,
                        model_loader=lambda path: calls.append(path) or _FakeModel(),
                    )
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
