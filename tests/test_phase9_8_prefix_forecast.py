from __future__ import annotations

import dataclasses
import json
import unittest
from functools import lru_cache
from pathlib import Path

from experiments.phase9_8_prefix_forecast import (
    FORECAST_COUNT,
    NativePrefixForecastContext,
    PARAMETER_RANGES,
    PROVIDER_VERSION,
    ROLLOUTS_PER_ACTION,
    _draw,
    _clip_probability,
    forecast,
    outcome_probabilities,
    select_action,
)


@lru_cache(maxsize=None)
def group_config(mechanism):
    table = json.loads(Path("results/phase9_8_configuration_table.json").read_text(encoding="utf-8"))
    return next(row for row in table["configurations"]
                if row["mechanism"] == mechanism and row["configuration_id"].endswith("cfg-20"))


def context(mechanism="transient_tool_failure", **overrides):
    config = group_config(mechanism)
    values = dict(scenario_id="scn-1", configuration_group_id=config["configuration_id"],
                  root_seed=20269804, mechanism=mechanism,
                  parameters=config["parameters"], susceptibility=0.4,
                  reversibility_proxy=0.65, trajectory_position=0.5,
                  disruption_history=0.5)
    values.update(overrides)
    return NativePrefixForecastContext.create(**values)


class Phase98PrefixForecastTests(unittest.TestCase):
    def test_exact_mechanism_equations_and_single_final_clipping(self):
        mechanisms = (
            "transient_tool_failure", "stale_contradictory_evidence", "irreversible_action",
            "late_trajectory_failure", "replanning_disruption",
        )
        for mechanism in mechanisms:
            with self.subTest(mechanism=mechanism):
                item = context(mechanism)
                p = dict(item.parameters)
                z, r, x, d = (item.susceptibility, item.reversibility_proxy,
                              item.trajectory_position, item.disruption_history)
                failure = 0.05 + 0.70 * z
                if mechanism == "transient_tool_failure":
                    repair, harm = p["repair_probability"], p["harm_probability"]
                elif mechanism == "stale_contradictory_evidence":
                    failure += p["failure_increment"]
                    repair, harm = p["repair_probability"], p["harm_probability"]
                elif mechanism == "irreversible_action":
                    severity = 1.0 - r
                    failure += p["failure_sensitivity"] * severity
                    repair = p["repair_probability"]
                    harm = p["harm_base"] + p["harm_sensitivity"] * severity
                elif mechanism == "late_trajectory_failure":
                    failure += p["failure_position_slope"] * x
                    repair = p["repair_intercept"] - p["repair_position_decline"] * x
                    harm = p["harm_intercept"] + p["harm_position_slope"] * x
                else:
                    failure += p["failure_disruption_slope"] * d
                    repair = p["repair_intercept"] - p["repair_disruption_decline"] * d
                    harm = p["harm_intercept"] + p["harm_disruption_slope"] * d
                expected = tuple(min(1.0, max(0.0, value)) for value in (failure, repair, harm))
                actual = outcome_probabilities(item)
                for value, expected in zip(actual, expected):
                    self.assertAlmostEqual(value, expected, places=14)
        self.assertEqual(_clip_probability(-0.4), 0.0)
        self.assertEqual(_clip_probability(1.4), 1.0)

    def test_context_rejects_invalid_parameter_bounds_and_features(self):
        config = group_config("transient_tool_failure")
        changed = dict(config["parameters"])
        changed["harm_probability"] += 0.001  # still within the global range, wrong for this group
        with self.assertRaisesRegex(ValueError, "exactly match frozen"):
            context(parameters=changed)
        with self.assertRaises(ValueError):
            context(susceptibility=float("nan"))
        with self.assertRaises(ValueError):
            context(trajectory_position=1.1)

    def test_context_rejects_unknown_group_and_mechanism_mismatch(self):
        with self.assertRaisesRegex(ValueError, "not in the frozen table"):
            context(configuration_group_id="made-up-group")
        transient = group_config("transient_tool_failure")
        with self.assertRaisesRegex(ValueError, "does not match frozen"):
            context(mechanism="stale_contradictory_evidence",
                    configuration_group_id=transient["configuration_id"])

    def test_parameter_bounds_match_frozen_configuration_table(self):
        table = json.loads(Path("results/phase9_8_configuration_table.json").read_text(encoding="utf-8"))
        frozen = table["generator"]["mechanisms_and_parameter_ranges"]
        self.assertEqual(
            PARAMETER_RANGES,
            {mechanism: {name: tuple(bounds) for name, bounds in parameters.items()}
             for mechanism, parameters in frozen.items()},
        )

    def test_context_is_immutable_and_excludes_evaluator_outcomes_and_draws(self):
        item = context()
        fields = {field.name for field in dataclasses.fields(item)}
        self.assertEqual(fields, {
            "scenario_id", "configuration_group_id", "root_seed", "mechanism", "parameters", "susceptibility",
            "reversibility_proxy", "trajectory_position", "disruption_history",
        })
        self.assertFalse(fields.intersection({
            "continuation_uniform", "repair_uniform", "harm_uniform", "label",
            "potential_outcomes", "realized_outcome", "verifier_result",
        }))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.scenario_id = "changed"

    def test_shared_continuation_and_domain_separated_intervention_draws(self):
        item = context()
        shared = _draw(item, 2, "continuation_shared_v1")
        self.assertEqual(shared, _draw(item, 2, "continuation_shared_v1"))
        repair = _draw(item, 2, "intervention_repair_v1", "intervene")
        harm = _draw(item, 2, "intervention_harm_v1", "intervene")
        continue_scoped_repair = _draw(item, 2, "intervention_repair_v1", "continue")
        self.assertNotEqual(shared, repair)
        self.assertNotEqual(repair, harm)
        self.assertNotEqual(repair, continue_scoped_repair)
        self.assertEqual(PROVIDER_VERSION, "phase9_8_native_prefix_forecast_v1")

    def test_forecast_is_deterministic_under_reordered_execution_and_exactly_eight(self):
        one, two = context(scenario_id="a"), context(scenario_id="b")
        baseline = forecast(one)
        _ = forecast(two)
        replay = forecast(one)
        self.assertEqual(baseline.continue_results, replay.continue_results)
        self.assertEqual(baseline.intervene_results, replay.intervene_results)
        self.assertEqual(baseline.selected_action, replay.selected_action)
        self.assertEqual(baseline.continue_pass_count, replay.continue_pass_count)
        self.assertEqual(baseline.intervene_pass_count, replay.intervene_pass_count)
        self.assertEqual(len(baseline.continue_results), 4)
        self.assertEqual(len(baseline.intervene_results), 4)
        self.assertEqual(baseline.forecast_count, FORECAST_COUNT)
        self.assertEqual(baseline.forecast_overhead_units, 8)

    def test_each_intervention_forecast_reuses_its_continuation_draw(self):
        item = context()
        failure, repair, harm = outcome_probabilities(item)
        expected_continue, expected_intervene = [], []
        for index in range(ROLLOUTS_PER_ACTION):
            continued = _draw(item, index, "continuation_shared_v1") >= failure
            repaired = _draw(item, index, "intervention_repair_v1", "intervene") < repair
            unharmed = _draw(item, index, "intervention_harm_v1", "intervene") >= harm
            expected_continue.append(continued)
            expected_intervene.append((continued or repaired) and unharmed)
        actual = forecast(item)
        self.assertEqual(actual.continue_results, tuple(expected_continue))
        self.assertEqual(actual.intervene_results, tuple(expected_intervene))

    def test_binary_cost_arithmetic_and_ties_continue(self):
        self.assertEqual(select_action(1.0, 0.999), "intervene")
        self.assertEqual(select_action(1.0, 1.0), "continue")
        self.assertEqual(select_action(0.999, 1.0), "continue")
        result = forecast(context())
        self.assertEqual(result.estimated_continue_cost, 3 * (1 - result.p_pass_continue))
        self.assertEqual(result.estimated_intervene_cost, 2 + 3 * (1 - result.p_pass_intervene))
        self.assertEqual(result.selected_action,
                         "intervene" if result.estimated_intervene_cost < result.estimated_continue_cost
                         else "continue")

    def test_diagnostics_separate_forecast_overhead_from_realized_cost(self):
        diagnostics = forecast(context()).to_diagnostics()
        self.assertEqual(diagnostics["forecast_count"], 8)
        self.assertEqual(diagnostics["forecast_overhead_units"], 8)
        self.assertTrue(diagnostics["overhead_is_separate_from_executed_task_steps_and_realized_task_cost"])
        self.assertNotIn("realized_task_cost", diagnostics)
        self.assertTrue(diagnostics["privileged_information"])


if __name__ == "__main__":
    unittest.main()
