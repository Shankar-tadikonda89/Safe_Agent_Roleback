"""Standalone v4 entrypoint; refuses execution without its approved v4 lock."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.phase9_8_pilot_factories import Phase98PilotFactories
from experiments.phase9_8_pilot_orchestrator_v4 import run_approved_native_pilot_v4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the approved Phase 9.8 wrapper-aware pilot")
    parser.add_argument("--approved-lock-sha256", required=True,
                        help="independently reviewed raw SHA-256 of the APPROVED v4 lock")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    factories = Phase98PilotFactories()
    result = run_approved_native_pilot_v4(
        approved_lock_sha256=args.approved_lock_sha256,
        scenario_factory=factories.scenarios,
        evaluator_outcome_factory=factories.outcomes,
        project_root=args.project_root,
    )
    print(f"pilot=COMPLETE scenarios={result.decision_count // 4} decisions={result.decision_count}")
    print(f"freeze={result.freeze_dir}")
    print(f"results={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
