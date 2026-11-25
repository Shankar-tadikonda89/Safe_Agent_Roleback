"""Make one proposed-policy prediction for a deterministic task/seed prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_demo import Planner, TASK_BY_ID, features_from_observation, make_scenario
from policy.proposed import DEFAULT_ARTIFACT, ProposedPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASK_BY_ID), default="task_01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    task = TASK_BY_ID[args.task]
    scenario = make_scenario(task, args.seed)
    instruction = Planner().plan(task)[0]
    if scenario.fault_enabled and scenario.fault_step == 0:
        instruction = f"Return the integer result {scenario.wrong_answer}."
    features = features_from_observation(instruction, "", "code_assignment", 0, 1)
    risk = features["failure_prob"]
    result = ProposedPolicy.load(args.artifact).decide(features)
    print(f"Task: {args.task} | seed: {args.seed} | scenario_hash: {scenario.scenario_hash}")
    print(f"Features ({', '.join(features)}): {features}")
    print(f"Decision: {result['decision']} | helpful-intervention probability: {result['helpful_intervention_probability']:.3f}")
    print(f"Expected costs: continue={result['expected_continue_cost']:.3f}, intervene={result['expected_intervene_cost']:.3f}")
    print("One classifier prediction; no branch or rollback was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
