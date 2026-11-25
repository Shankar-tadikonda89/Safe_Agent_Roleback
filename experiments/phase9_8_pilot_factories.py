"""Frozen Phase 9.8 pilot factories with a delayed evaluator-outcome callback.

Scenario generation creates public Phase 9.7 observations and keeps evaluator
draws private in this factory instance. Potential outcomes are not calculated
until the orchestrator supplies a verified complete decision freeze.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from experiments import phase9_7_simulator as sim
from experiments.phase9_8_decision_freeze import FrozenPilotDesign, VerifiedDecisionFreeze
from experiments.phase9_8_pilot_runner import (
    COMPARATIVE_POLICIES, PilotScenario, _validate_complete_decision_set,
)
from experiments.phase9_8_prefix_forecast import NativePrefixForecastContext


class PilotFactoryError(ValueError):
    """Frozen pilot factory provenance or evaluator boundary failed."""


@dataclass
class Phase98PilotFactories:
    """Callbacks used by the approved orchestrator; no callback is a CLI."""

    _generated: dict[str, sim.GeneratedScenario] = field(default_factory=dict, init=False, repr=False)
    _configurations: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)

    def scenarios(
        self, design: FrozenPilotDesign,
    ) -> tuple[Sequence[PilotScenario], Mapping[str, NativePrefixForecastContext]]:
        if self._generated:
            raise PilotFactoryError("scenario factory is single-use")
        scenarios: list[PilotScenario] = []
        contexts: dict[str, NativePrefixForecastContext] = {}
        for scenario_id, (group_id, mechanism, root_seed, scenario_seed) in sorted(
            design.scenario_specs.items(), key=lambda item: (item[1][0], item[1][3]),
        ):
            if design.configuration_mechanisms.get(group_id) != mechanism:
                raise PilotFactoryError("scenario configuration does not match frozen membership")
            parameters = dict(design.configuration_parameters[group_id])
            generated = sim.build_scenario(
                root_seed=root_seed, scenario_seed=scenario_seed, mechanism=mechanism,
            )
            if generated.policy.scenario_id != scenario_id or generated.evaluator.scenario_id != scenario_id:
                raise PilotFactoryError("simulator output identity differs from frozen manifest")
            features = sim.extract_features(generated.policy.snapshot.observation)
            scenario = PilotScenario(
                policy_scenario=generated.policy,
                configuration_group_id=group_id,
                mechanism=mechanism,
                root_seed=root_seed,
                scenario_seed=scenario_seed,
            )
            context = NativePrefixForecastContext.create(
                scenario_id=scenario_id,
                configuration_group_id=group_id,
                root_seed=root_seed,
                mechanism=mechanism,
                parameters=parameters,
                susceptibility=generated.evaluator.susceptibility,
                reversibility_proxy=features.reversibility_proxy,
                trajectory_position=features.trajectory_position,
                disruption_history=features.disruption_history,
            )
            scenarios.append(scenario)
            contexts[scenario_id] = context
            self._generated[scenario_id] = generated
            self._configurations[scenario_id] = {
                "mechanism": mechanism, "parameters": parameters,
            }
        design.validate_scenarios(scenarios)
        return tuple(scenarios), contexts

    def outcomes(
        self, scenarios: Sequence[PilotScenario], freeze: VerifiedDecisionFreeze,
    ) -> Mapping[str, sim.PotentialOutcomes]:
        if not isinstance(freeze, VerifiedDecisionFreeze):
            raise PilotFactoryError("evaluator outcomes require a verified persisted decision freeze")
        rows = list(scenarios)
        ids = {row.policy_scenario.scenario_id for row in rows}
        if ids != set(self._generated) or len(ids) != len(rows):
            raise PilotFactoryError("outcome request does not match generated pilot scenarios")
        try:
            _validate_complete_decision_set(freeze.decisions, rows)
        except (TypeError, ValueError) as exc:
            raise PilotFactoryError("verified freeze lacks complete matched decisions") from exc
        if {decision.policy for decision in freeze.decisions} != set(COMPARATIVE_POLICIES):
            raise PilotFactoryError("verified freeze does not use the amended four-policy set")

        # Importing the pure evaluator function does not generate anything;
        # this loop is the first point where potential outcomes are computed.
        from experiments.phase9_8_training import configured_potential_outcomes

        results: dict[str, sim.PotentialOutcomes] = {}
        for scenario in rows:
            scenario_id = scenario.policy_scenario.scenario_id
            generated = self._generated[scenario_id]
            results[scenario_id] = configured_potential_outcomes(
                generated.policy,
                generated.evaluator,
                self._configurations[scenario_id],
            )
        return results
