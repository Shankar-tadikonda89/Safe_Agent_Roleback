from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import experiments.phase9_8_native_prefix_loader as loader
from experiments.phase9_8_adapter import (
    METADATA_PATH,
    MODEL_PATH,
    Phase98ArtifactError,
    _sha256,
)
class _FakeModel:
    def __init__(self):
        scaler_type = type("StandardScaler", (), {})
        classifier_type = type("LogisticRegression", (), {})
        scaler = scaler_type()
        scaler.n_features_in_ = 4
        classifier = classifier_type()
        classifier.classes_ = np.asarray([0, 1, 2, 3])
        classifier.C = 1.0
        self.named_steps = {"scaler": scaler, "classifier": classifier}
        self.n_features_in_ = 4

    def predict_proba(self, _features):
        return np.asarray([[0.1, 0.2, 0.3, 0.4]])


class Phase98NativePrefixLoaderTests(unittest.TestCase):
    def _fixture(self):
        temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        root = Path(temp.name)
        source_root = Path.cwd()
        for relative in loader._BOUND_PATHS.values():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((source_root / relative).read_bytes())
        return temp, root

    def _write_approved_lock(self, root: Path) -> Path:
        lock = json.loads((Path.cwd() / loader.NATIVE_LOCK_PATH).read_text(encoding="utf-8"))
        # This approval state exists only in a temporary test fixture. The
        # project lock remains DRAFT_NOT_APPROVED and is never changed.
        lock["approval_status"] = "APPROVED"
        lock["approval_note"] = "Test-only fixture; no production approval."
        # Bind the current bytes copied into this temporary fixture. This also
        # ensures source/addendum edits fail on their actual hash gate instead
        # of making the positive fake-loader fixture stale.
        lock["sha256"] = {
            name: _sha256(root / relative)
            for name, relative in loader._BOUND_PATHS.items()
        }
        lock.pop("lock_sha256", None)
        lock["lock_sha256"] = hashlib.sha256(
            json.dumps(lock, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        path = root / loader.NATIVE_LOCK_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lock, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    def _assert_rejected_before_loader(self, root: Path, lock_path: Path, *,
                                       approved_hash: str | None = None) -> None:
        calls = []
        with self.assertRaises(Phase98ArtifactError):
            loader.load_native_prefix_adapter(
                root=root, lock_path=lock_path,
                approved_lock_sha256=approved_hash or _sha256(lock_path),
                model_loader=lambda path: calls.append(path) or _FakeModel(),
            )
        self.assertEqual(calls, [])

    def test_approved_fixture_validates_all_gates_before_fake_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            calls = []
            adapter = loader.load_native_prefix_adapter(
                root=root, lock_path=lock_path,
                approved_lock_sha256=_sha256(lock_path),
                model_loader=lambda path: calls.append(path) or _FakeModel(),
            )
            self.assertEqual(calls, [root / MODEL_PATH])
            self.assertEqual(adapter.artifact_sha256, json.loads(
                (root / METADATA_PATH).read_text(encoding="utf-8")
            )["artifact_sha256"])

    def test_draft_lock_is_rejected_before_loader(self):
        lock_path = Path(loader.NATIVE_LOCK_PATH)
        self._assert_rejected_before_loader(Path.cwd(), lock_path)

    def test_wrong_independent_raw_lock_hash_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            self._assert_rejected_before_loader(root, lock_path, approved_hash="0" * 64)

    def test_canonical_lock_tampering_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["forecast_count_per_decision"] = 7
            lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
            self._assert_rejected_before_loader(root, lock_path)

    def test_schema_policy_and_loader_version_mismatches_reject_before_loader(self):
        for field, value in (
            ("schema_version", "wrong-schema"),
            ("comparator_version", "wrong-comparator"),
            ("loader_version", "wrong-loader"),
            ("frozen_feature_order", ["reversed", "feature", "schema", "order"]),
        ):
            temp, root = self._fixture()
            with temp, self.subTest(field=field):
                lock_path = self._write_approved_lock(root)
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                lock[field] = value
                lock.pop("lock_sha256", None)
                lock["lock_sha256"] = hashlib.sha256(
                    json.dumps(lock, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
                ).hexdigest()
                lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
                self._assert_rejected_before_loader(root, lock_path)

    def test_bound_artifact_hash_mismatch_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            addendum = root / loader.NATIVE_ADDENDUM_PATH
            addendum.write_text(addendum.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            self._assert_rejected_before_loader(root, lock_path)

    def test_inherited_hash_mismatch_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["inherited_frozen_raw_hashes"]["model"] = "0" * 64
            lock.pop("lock_sha256", None)
            lock["lock_sha256"] = hashlib.sha256(
                json.dumps(lock, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()
            lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
            self._assert_rejected_before_loader(root, lock_path)

    def test_pilot_partition_mismatch_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["pilot_scope"]["scenario_count"] = 1
            lock.pop("lock_sha256", None)
            lock["lock_sha256"] = hashlib.sha256(
                json.dumps(lock, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()
            lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
            self._assert_rejected_before_loader(root, lock_path)

    def test_metadata_validation_failure_is_rejected_before_loader(self):
        temp, root = self._fixture()
        with temp:
            lock_path = self._write_approved_lock(root)
            with patch.object(loader, "validate_metadata", side_effect=Phase98ArtifactError("invalid fixture")):
                self._assert_rejected_before_loader(root, lock_path)


if __name__ == "__main__":
    unittest.main()
