from __future__ import annotations

import unittest

from experiments.phase9_8_pilot_factories import Phase98PilotFactories, PilotFactoryError


class PilotFactoryBoundaryTests(unittest.TestCase):
    def test_evaluator_factory_requires_verified_freeze_before_any_outcome_work(self):
        factory = Phase98PilotFactories()
        with self.assertRaises(PilotFactoryError):
            factory.outcomes((), object())  # no scenario builder or outcome function is reached
        self.assertEqual(factory._generated, {})

    def test_pilot_factory_callbacks_are_separate(self):
        factory = Phase98PilotFactories()
        self.assertTrue(callable(factory.scenarios))
        self.assertTrue(callable(factory.outcomes))
        self.assertEqual(factory._generated, {})


if __name__ == "__main__":
    unittest.main()
