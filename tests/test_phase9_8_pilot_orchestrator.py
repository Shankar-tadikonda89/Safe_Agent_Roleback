from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from experiments.phase9_8_native_pilot_loader_v3 import NATIVE_PILOT_LOCK_V3_PATH
from experiments.phase9_8_pilot_orchestrator import run_approved_native_pilot
from experiments.phase9_8_native_pilot_loader_v3 import NativePilotV3ArtifactError


class PilotOrchestratorGateTests(unittest.TestCase):
    def test_draft_lock_stops_before_scenario_or_evaluator_factories(self):
        lock = Path(NATIVE_PILOT_LOCK_V3_PATH)
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        calls = []

        def scenarios(_design):
            calls.append("scenario")
            raise AssertionError("scenario generation must not start with a draft lock")

        def outcomes(_scenarios, _freeze):
            calls.append("outcomes")
            raise AssertionError("evaluator outcomes must not be accessed")

        with self.assertRaises(NativePilotV3ArtifactError):
            run_approved_native_pilot(
                approved_lock_sha256=digest,
                scenario_factory=scenarios,
                evaluator_outcome_factory=outcomes,
                project_root=Path.cwd(),
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
