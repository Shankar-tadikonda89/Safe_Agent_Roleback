from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.phase9_5_observations import FEATURE_ORDER
from experiments.phase9_7_runner import OUTCOME_CLASSES
from experiments.phase9_8_adapter import (
    ADAPTER_VERSION,
    CONFIGURATION_PATH,
    LOCK_PATH,
    METADATA_PATH,
    MODEL_PATH,
    PILOT_PREREG_PATH,
    PROTOCOL_PATH,
    SPLIT_PATH,
    TRAINING_REPORT_PATH,
    _LOAD_GATE_TOKEN,
    Phase98ArtifactError,
    Phase98ProposedPolicyAdapter,
    _canonical_bytes,
    _metadata_content_hash,
    _sha256,
    load_phase98_adapter,
    validate_metadata,
)


class _FakeModel:
    def __init__(self, probabilities=(0.1, 0.2, 0.3, 0.4), classes=(0, 1, 2, 3)):
        self.output = np.asarray([probabilities])
        scaler_type = type("StandardScaler", (), {})
        classifier_type = type("LogisticRegression", (), {})
        scaler = scaler_type()
        scaler.n_features_in_ = 4
        classifier = classifier_type()
        classifier.classes_ = np.asarray(classes)
        classifier.C = 1.0
        self.named_steps = {"scaler": scaler, "classifier": classifier}
        self.n_features_in_ = 4
        self.calls = []

    def predict_proba(self, values):
        self.calls.append(np.asarray(values).copy())
        return self.output


def _metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _feature_row():
    return dict(zip(FEATURE_ORDER, (0.2, 0.65, 0.5, 0.0), strict=True))


class Phase98AdapterTests(unittest.TestCase):
    def test_metadata_contract_and_phase97_separation(self):
        metadata = _metadata()
        validate_metadata(metadata)
        self.assertEqual(metadata["artifact_version"], "phase9.8-multinomial-cost-sensitive-v1")
        self.assertNotEqual(metadata["artifact_version"], "phase9-7-policy-contract-v1")
        self.assertEqual(tuple(metadata["feature_names"]), FEATURE_ORDER)
        self.assertEqual(tuple(metadata["class_names"]), OUTCOME_CLASSES)

    def test_metadata_rejects_order_hash_cost_or_selection_mutation(self):
        for key, value in (
            ("feature_names", list(reversed(FEATURE_ORDER))),
            ("class_names", list(reversed(OUTCOME_CLASSES))),
            ("selected_C", 10.0),
            ("protocol_sha256", "0" * 64),
        ):
            metadata = _metadata()
            metadata[key] = value
            with self.subTest(key=key), self.assertRaises(Phase98ArtifactError):
                validate_metadata(metadata)

    def test_adapter_calls_frozen_classifier_once_and_preserves_order(self):
        metadata = _metadata()
        model = _FakeModel()
        adapter = Phase98ProposedPolicyAdapter(
            model, metadata, artifact_sha256=metadata["artifact_sha256"], _gate_token=_LOAD_GATE_TOKEN
        )
        result = adapter.decide(_feature_row())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0].shape, (1, 4))
        self.assertEqual(result["adapter_version"], ADAPTER_VERSION)
        self.assertEqual(tuple(result["class_probabilities"]), OUTCOME_CLASSES)
        self.assertIn(result["selected_action"], {"continue", "intervene"})

    def test_tie_continues_and_expected_cost_uses_frozen_matrix(self):
        metadata = _metadata()
        model = _FakeModel((2 / 3, 1 / 3, 0.0, 0.0))
        adapter = Phase98ProposedPolicyAdapter(model, metadata, artifact_sha256=metadata["artifact_sha256"], _gate_token=_LOAD_GATE_TOKEN)
        result = adapter.decide(_feature_row())
        self.assertAlmostEqual(result["expected_continue_cost"], 2.0)
        self.assertAlmostEqual(result["expected_intervene_cost"], 2.0)
        self.assertEqual(result["selected_action"], "continue")

    def test_invalid_features_and_probability_outputs_fail_closed(self):
        metadata = _metadata()
        model = _FakeModel((0.1, 0.2, 0.3, 0.4))
        adapter = Phase98ProposedPolicyAdapter(model, metadata, artifact_sha256=metadata["artifact_sha256"], _gate_token=_LOAD_GATE_TOKEN)
        for invalid in (
            {**_feature_row(), "failure_prob": "0.2"},
            {**_feature_row(), "failure_prob": True},
            {**_feature_row(), "failure_prob": float("nan")},
            {**_feature_row(), "failure_prob": 1.1},
            {"reversibility_proxy": 0.5},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adapter.decide(invalid)
        for probabilities in ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3, float("nan")),
                              (0.1, 0.2, 0.3, 0.5), ("0.1", 0.2, 0.3, 0.4)):
            model.output = np.asarray([probabilities], dtype=object)
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                adapter.decide(_feature_row())

    def test_bad_model_class_order_is_rejected(self):
        metadata = _metadata()
        with self.assertRaises(Phase98ArtifactError):
            Phase98ProposedPolicyAdapter(
                _FakeModel(classes=(1, 0, 2, 3)), metadata,
                artifact_sha256=metadata["artifact_sha256"], _gate_token=_LOAD_GATE_TOKEN,
            )

    def test_draft_lock_refuses_before_deserialization(self):
        lock_path = Path(LOCK_PATH)
        if not lock_path.is_file():
            self.skipTest("draft pilot lock has not yet been created")
        called = []
        with self.assertRaises(Phase98ArtifactError):
            load_phase98_adapter(
                lock_path=lock_path,
                approved_lock_sha256=_sha256(lock_path),
                model_loader=lambda path: called.append(path),
            )
        self.assertEqual(called, [])

    def test_adapter_cannot_be_constructed_outside_gated_loader(self):
        metadata = _metadata()
        with self.assertRaises(Phase98ArtifactError):
            Phase98ProposedPolicyAdapter(
                _FakeModel(), metadata, artifact_sha256=metadata["artifact_sha256"]
            )

    def test_approved_fixture_lock_checks_hashes_before_injected_loader(self):
        fake_model = _FakeModel()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for path in (PROTOCOL_PATH, CONFIGURATION_PATH, SPLIT_PATH, PILOT_PREREG_PATH):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
            model_path = root / MODEL_PATH
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"test fixture bytes; not a serialized model")
            for path in (TRAINING_REPORT_PATH,):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fixture report")

            metadata = _metadata()
            metadata["artifact_sha256"] = _sha256(model_path)
            metadata["model_variants"]["full_features"]["sha256"] = _sha256(model_path)
            metadata["metadata_sha256"] = _metadata_content_hash(metadata)
            metadata_path = root / METADATA_PATH
            metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            hashes = {
                "protocol": _sha256(root / PROTOCOL_PATH),
                "configuration_table": _sha256(root / CONFIGURATION_PATH),
                "split_manifest": _sha256(root / SPLIT_PATH),
                "model": _sha256(model_path),
                "metadata": _sha256(metadata_path),
                "training_report": _sha256(root / TRAINING_REPORT_PATH),
                "pilot_preregistration": _sha256(root / PILOT_PREREG_PATH),
            }
            lock = {
                "schema_version": "phase9-8-independent-pilot-lock-v1",
                "approval_status": "APPROVED",
                "adapter_version": ADAPTER_VERSION,
                "sha256": hashes,
                "canonical_design_hashes": {
                    "configuration_table_sha256": "efc4caf9cf570038a772bb76df57215cc412e3fce0e207566119db6dd83a3963",
                    "split_manifest_sha256": "9686e7d0a53b5ec261afd9f02cca4d3c80bd036e4d6a55399aaf19848c1eda99",
                    "metadata_content_sha256": metadata["metadata_sha256"],
                },
                "pilot_partition": {
                    "root_seed": 20269804,
                    "configuration_group_ids": [
                        group["configuration_id"] for group in json.loads((root / SPLIT_PATH).read_text())["groups"]
                        if group["partition"] == "pilot"
                    ],
                    "scenario_count": 2560,
                },
                "metadata_content_sha256": metadata["metadata_sha256"],
                "selected_C": 1.0,
                "feature_order": list(FEATURE_ORDER),
                "class_order": list(OUTCOME_CLASSES),
            }
            lock["lock_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
            lock_path = root / "results/phase9_8_pilot_lock.json"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
            invoked = []
            adapter = load_phase98_adapter(
                root=root, lock_path=lock_path,
                approved_lock_sha256=_sha256(lock_path),
                model_loader=lambda path: invoked.append(path) or fake_model,
            )
            self.assertEqual(len(invoked), 1)
            self.assertEqual(adapter.decide(_feature_row())["selected_action"], "continue")


if __name__ == "__main__":
    unittest.main()
