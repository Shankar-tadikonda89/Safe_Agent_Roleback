# Phase 9.8 Post-Pilot Audit

**Audit type:** Read-only validation of the completed pilot and its preserved records  
**Run:** `run-3d4f75d0ae811b9a7342`  
**Decision freeze:** `freeze-3d4f75d0ae811b9a7342`

## Executive result

The approved v4 lock, its 45 bound-file hashes, the decision-freeze record digest, the freeze manifest's canonical digest, matched policy coverage, and the scored-run metrics all verified. Recalculated metrics exactly match `comparative_metrics.json`. The primary paired cost estimate and its configured bootstrap interval also reproduce from the scored JSONL.

No numerical discrepancy was found in the reported policy table. Two reporting limitations remain: the metrics JSON does not contain the primary estimate or interval, and the pilot did not persist a complete scenario-level potential-outcomes file. The summaries can be recomputed from scored runs, but the entire set of evaluator counterfactuals cannot be independently replayed from a preserved outcome log. The original protocol and pilot preregistration still contain superseded text/status labels; the v3 supersession and v4 addendum must accompany them in a submission package.

## Integrity and coverage checks

| Check | Result | Evidence |
|---|---|---|
| v4 lock approval and raw digest | **PASS** | Lock status `APPROVED`; raw SHA-256 `893a212550183db3315858cc9cd46210674c548da664f6f5a1d2007d26fabd72`. |
| Canonical lock integrity | **PASS** | Canonical SHA-256 `172663ff409325929e4d3b804fa33cab407bb4fcf9a41ed935ae340f0b513dff`, equal to the recorded canonical field. |
| Bound-file integrity | **PASS** | All 45 current files in `experiments.phase9_8_native_pilot_loader_v4._actual_hashes_v4` match `bound_file_sha256`. |
| Decision record integrity | **PASS** | `decisions.jsonl` SHA-256 `3d4f75d0ae811b9a73423c74fb592b33d7dca4809aa7c15ba212c0c9f761e06f` matches the freeze manifest. The manifest canonical hash recomputes. |
| Freeze/run identifiers | **PASS** | Both directories use the same record-hash prefix: `3d4f75d0ae811b9a7342`. |
| Scenario and decision counts | **PASS** | 2,560 unique pilot scenarios; 10,240 frozen decisions and 10,240 scored rows; exactly 2,560 records per each of the four policies. |
| Matched scenario provenance | **PASS** | For every scenario, the four policies share configuration group, mechanism, root/scenario seeds, initial-state hash, decision sequence, features, and selected decision across the freeze-to-score join. Ten pilot configuration groups were present: two per mechanism. Root seed was `20269804`. |
| Counterfactual application semantics | **PASS** | All scored rows state `confirmed_application=false`, `intervention_applied=null`, and `one_decision_counterfactual_simulation`. |
| `pilot_test` exclusion | **PASS** | The separate descriptive record says `comparative_metric_eligible=false`; `pilot_test` is absent from `comparative_metrics.json`. |

The freeze and run records were read directly from:

- `results/phase9_8_pilot/decision_freezes/freeze-3d4f75d0ae811b9a7342/decisions.jsonl`
- `results/phase9_8_pilot/decision_freezes/freeze-3d4f75d0ae811b9a7342/freeze_manifest.json`
- `results/phase9_8_native_pilot_v4/runs/run-3d4f75d0ae811b9a7342/comparative_scored_runs.jsonl`
- `results/phase9_8_native_pilot_v4/runs/run-3d4f75d0ae811b9a7342/comparative_metrics.json`
- `results/phase9_8_native_pilot_v4/runs/run-3d4f75d0ae811b9a7342/pilot_test_descriptive_selection.json`

## Recomputed results

`experiments.phase9_8_pilot_runner.calculate_pilot_metrics` was applied to the existing scored JSONL. Its output matched the stored policy metrics exactly.

| Policy | Verifier passes | Success rate | Mean realized cost | Mean steps | Selected `intervene` actions |
|---|---:|---:|---:|---:|---:|
| `no_intervention` | 1,410 / 2,560 | 55.08% | 1.34765625 | 1.000000 | 0 |
| `naive_threshold` | 1,083 / 2,560 | 42.30% | 1.730859375 | 0.640625 | 0 |
| `proposed` | 1,410 / 2,560 | 55.08% | 1.34765625 | 1.000000 | 0 |
| `phase9_8_native_prefix_forecast_v1` | 1,514 / 2,560 | 59.14% | 1.40703125 | 1.090625 | 232 |

The naive threshold policy aborted 920 scenarios; aborts are failures and are not counted as interventions. The prefix comparator recorded 108 recovery, 83 unnecessary-intervention, 37 ineffective-intervention, and 4 intervention-harm outcomes among its 232 selected intervention actions. These are simulator counterfactual outcomes, not confirmed real interventions. Its selected-intervention rate is 232/2,560 = 9.06%; its harm rate among selected actions is 4/232 = 1.72%. Recovery and unnecessary counts use the frozen denominators of 1,150 continuation failures and 1,410 continuation passes, respectively.

The separate `pilot_test_descriptive_selection.json` records `continue`, a continuation mean cost of 1.34765625, an intervention mean cost of 2.655078125, and `comparative_metric_eligible=false`. This outcome-selected descriptive choice is not a fifth comparative result.

### Primary paired cost comparison

Recomputed using `experiments.phase9_8_pilot_runner.primary_paired_cost_estimand` from the scored records:

- Estimand: proposed minus no-intervention mean realized task cost.
- Estimate: **0.000000** over 2,560 matched scenarios and 10 pilot configuration groups.
- Paired mechanism-stratified percentile interval: **[0.000000, 0.000000]**.
- Bootstrap: 10,000 resamples, seed `20269810`, resampling configuration-group means within mechanisms.

The point estimate and interval are zero because `proposed` selected `continue` in all 2,560 scenarios, exactly matching `no_intervention`. The interval is descriptive, not a significance test, and not evidence of policy equivalence beyond these recorded synthetic scenarios.

## Discrepancies and reproducibility limits

1. **No metric discrepancy.** All listed counts, rates, mean costs, mean steps, intervention counts, and the primary estimate/interval reproduce from the recorded rows.
2. **Primary interval is not embedded in the metrics JSON.** The stored JSON contains per-policy metrics only. The interval above was recomputed from the scored JSONL using the frozen helper and seed; submission materials should identify it as a recomputation rather than a field copied from `comparative_metrics.json`.
3. **No complete evaluator potential-outcomes JSONL exists in the run directory.** The orchestrator computes evaluator outcomes after the persisted decision freeze and passes them to scoring. Scored rows preserve the selected outcome information needed for reported metrics, but do not preserve every unselected intervention potential for every scenario. The `pilot_test` summary is preserved, but its per-scenario selection inputs cannot be reconstructed from that summary alone.
4. **Metric reproducibility is verified; full experiment replay is not.** This audit recalculated summaries and the bootstrap from saved records. It did not rerun scenario generation or the pilot. Deterministic seeds and source code are preserved, but repeat-run identity was not tested here.
5. **Supersession documents must travel with the report.** The original `results/phase9_8_protocol.md` and `results/phase9_8_pilot_preregistration.md` still show the old prefix-comparator name/formula or draft status. `results/phase9_8_pilot_preregistration_supersession_v3.md`, bound by the approved lock, supersedes the conflicting comparator definition; the v4 wrapper-loader addendum records the artifact-format fix. The v3/v4 addendum headers themselves still display `DRAFT_NOT_APPROVED` even though the corresponding lock approval/run was explicitly authorized. This stale status wording should be addressed in submission documentation; it was not edited during this read-only audit.
6. **No held-out evaluation occurred.** These are pilot-partition results (10 groups, two per mechanism), not results on the 40-group held-out partition. They make no formal power, superiority, calibration, or real-world effectiveness claim.

## Source files used

- `results/phase9_8_native_pilot_lock_v4.json` and `experiments/phase9_8_native_pilot_loader_v4.py` — lock approval, canonical digest, and 45 bound files.
- `experiments/phase9_8_decision_freeze.py` — `_coverage_payload`, `_record_bytes`, and `_verify_decision_freeze` schemas and integrity rules.
- `experiments/phase9_8_pilot_runner.py` — `_validate_complete_decision_set`, `score_frozen_decisions`, `calculate_pilot_metrics`, `primary_paired_cost_estimand`, `sim_cost`, and `select_pilot_test_action`.
- `experiments/phase9_8_pilot_orchestrator_v4.py` — loader → decisions → persisted freeze/reread → evaluator callback → scored records → metrics/selection order; output paths.
- `experiments/phase9_8_pilot_factories.py` and `experiments/phase9_8_training.py` — scenario/evaluator separation and `configured_potential_outcomes`.
- `experiments/phase9_8_adapter.py` and `experiments/phase9_7_runner.py` — four-class expected-cost calculation and frozen matrix.
- `experiments/phase9_8_prefix_forecast.py` — the Phase 9.8-native eight-forecast comparator.
- `experiments/phase9_5_observations.py` and `experiments/phase9_5_event_history.py` — decision-time feature semantics and the uncalibrated synthetic risk score.
- `results/phase9_8_protocol.md`, `results/phase9_8_pilot_preregistration_supersession_v3.md`, and `results/phase9_8_pilot_wrapper_loader_addendum_v4.md` — frozen design, supersession, and wrapper-loader context.
- `Codex_Self_Contained_Final_Build_Spec_Safe_Intervention.md` and `README.md` — project objective and previously documented scope.

No tests, experiments, model loads, network calls, or file modifications outside this new report directory were performed for the audit.
