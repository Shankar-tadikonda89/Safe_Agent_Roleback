# Safe Intervention Without Rollback
## Final Phase 9.8 Pilot Research Report

### Abstract

This report evaluates a learned four-class intervention policy in a frozen, offline synthetic pilot. Four policies were compared on 2,560 matched scenarios drawn from ten configuration groups spanning five simulator mechanisms. The proposed policy selected `continue` in every scenario. Its success rate and mean realized cost therefore exactly matched the no-intervention baseline: 55.08% and 1.34765625 cost units per scenario. The primary paired cost difference was 0.0, with a mechanism-stratified configuration-group bootstrap interval of [0.0, 0.0]. The naive threshold policy aborted 920 scenarios and had lower success and higher mean cost. A simulator-assisted prefix comparator selected intervention in 232 scenarios and had the highest observed success rate, but also a higher mean cost than no-intervention; it had privileged simulator information and unproven compute matching. These are counterfactual synthetic results. No intervention was confirmed as applied, no held-out partition was evaluated, and the results do not establish real-world LLM-agent effectiveness, policy superiority, or statistical significance.

## 1. Research problem

The capstone asks whether a lightweight, single-pass decision policy can use inexpensive trajectory observations to decide when intervention is warranted, reducing unnecessary disruption without counterfactual re-execution at decision time. The central research idea is a cost-sensitive action choice between continuing and intervening, placed between a risk signal and an agent executor. The authoritative specification frames this as a research objective, not as a claim that a synthetic simulation proves operational benefit (`Codex_Self_Contained_Final_Build_Spec_Safe_Intervention.md`, §1).

The Phase 9.8 pilot tests the decision and evaluator machinery under specified synthetic mechanisms. It does not run live Planner/Coder/Verifier agents or call Gemini. Its intervention branches are counterfactual simulator outcomes; it is not an executed repair-and-replan trial.

## 2. Decision observations and proposed policy

The saved Phase 9.8 classifier consumes the frozen feature order:
`failure_prob`, `reversibility_proxy`, `trajectory_position`, and `disruption_history`.

- **`failure_prob`:** despite its legacy name, it is an explicitly uncalibrated synthetic risk score. It is a weighted visible-evidence score: one quarter each for anomaly score, anomaly alert, a capped visible-error component, and a capped visible-warning component. Its stated target is rejection at the next verifier checkpoint, or final verification when no intermediate checkpoint exists. It is not a calibrated real-world failure probability (`experiments/phase9_5_observations.py`, `score_visible_evidence`).
- **`reversibility_proxy`:** a fixed metadata mapping: `read_only=0.95`, `reversible_write=0.65`, `irreversible_write=0.20`. It is a proxy, not a measured real-world reversibility probability.
- **`trajectory_position`:** zero-based current plan position divided by `planned_step_count - 1`; a single-step plan maps to zero.
- **`disruption_history`:** prior `intervention_applied` events from the complete ledger prefix, normalized as `min(count/3, 1)`. The current decision and later events are excluded (`experiments/phase9_5_event_history.py`, `reduce_disruption_history`).

The proposed model is a `StandardScaler` followed by multinomial scikit-learn `LogisticRegression`. It was trained on the frozen fit partition, and `C=1.0` was selected from `{0.1, 1, 10}` using validation multiclass log loss and the frozen lower-C tie rule. There is no post-hoc calibrator. At inference the model produces probabilities over four outcome classes; the policy applies the fixed cost matrix and intervenes only when intervention has strictly lower expected cost. The model was trained by maximum likelihood; the cost sensitivity enters in the inference decision rule (`experiments/phase9_8_training.py`, `train_and_select`; `experiments/phase9_8_adapter.py`, `expected_costs`).

### Frozen potential-outcome costs

| Potential-outcome class | Continue | Intervene |
|---|---:|---:|
| `recovery` | 3 | 2 |
| `unnecessary_intervention` | 0 | 2 |
| `ineffective_intervention` | 3 | 5 |
| `intervention_harm` | 0 | 5 |

For model class probabilities `(q_recovery, q_unnecessary, q_ineffective, q_harm)`, the expected costs are:

```text
E_continue  = 3 * (q_recovery + q_ineffective)
E_intervene = 2 + 3 * (q_ineffective + q_harm)
```

Choose the lower-cost action; ties continue. The policy makes one classifier prediction per decision and does not branch or roll back at inference.

## 3. Experimental setup

The frozen design contains 120 configuration groups across five mechanism families: transient tool failure, stale/contradictory evidence, irreversible action, late trajectory failure, and replanning disruption. Each group contains 256 scenario seeds. The pilot used ten groups—two per mechanism—and 2,560 scenarios under root seed `20269804`. The remaining 40 held-out groups were not generated or evaluated. The design is exploratory, and the configuration groups—not individual scenarios—are the independent uncertainty units (`results/phase9_8_protocol.md`, §§1–3; `results/phase9_8_split_manifest.json`).

The four comparative policies were:

1. `no_intervention`: always continue.
2. `naive_threshold`: abort when the synthetic risk score is strictly greater than 0.5. Abort remains a failure and is not counted as an intervention.
3. `proposed`: use the saved four-class model and frozen expected-cost rule.
4. `phase9_8_native_prefix_forecast_v1`: use four forecasts per action (eight total) and action-specific pass estimates:

   ```text
   C_continue  = 3 * (1 - p_pass_continue)
   C_intervene = 2 + 3 * (1 - p_pass_intervene)
   ```

   Ties continue. This comparator is Phase 9.8-native, not an exact Phase 9.2 port. It uses privileged simulator susceptibility/configuration information unavailable to the proposed policy, and equal compute has not been established (`results/phase9_8_pilot_preregistration_supersession_v3.md`).

All policies were matched on scenario identity, seeds, configuration group, and initial-state hash. Decisions were persisted and verified against a decision-freeze manifest before evaluator outcomes were joined (`experiments/phase9_8_pilot_orchestrator_v4.py`, `run_approved_native_pilot_v4`; `experiments/phase9_8_decision_freeze.py`).

The fifth policy, `pilot_test`, selects one global action using pilot outcomes. It was stored only in a separate descriptive record after the comparative decision freeze and excluded from comparative metrics. It selected `continue`; its in-sample continuation and intervention mean costs were 1.34765625 and 2.655078125, respectively. This outcome-selected statistic is not a fair comparative estimate.

### One-decision counterfactual semantics

For every scenario, policies select `continue`, `intervene`, or `abort`; the evaluator then uses paired potential outcomes. No repair, tool operation, plan mutation, or replan is actually executed by this pilot. The records identify selected actions separately from confirmed application; all `confirmed_application` fields are false and all `intervention_applied` fields are null. Therefore “intervention count” means selected intervention decisions, and the reported disruption rate is a selected-action proxy. Recovery and harm counts are simulated potential outcomes, not observed operational effects. This pilot does not test an applied intervention or a real no-rollback workflow (`results/phase9_8_pilot_preregistration_supersession_v3.md`, §Pilot execution semantics; `experiments/phase9_8_pilot_runner.py`, `counterfactual_application_record` and `score_frozen_decisions`).

## 4. Verified results

The table below is recomputed from the recorded scored JSONL. Success is final verifier passes divided by 2,560 attempts. Mean realized cost and mean steps are arithmetic means across attempts. Intervention counts are selected `intervene` actions, not confirmed application.

| Policy | Success | Mean realized cost | Mean steps | Selected interventions |
|---|---:|---:|---:|---:|
| `no_intervention` | 1,410/2,560 (55.08%) | 1.34765625 | 1.000000 | 0 |
| `naive_threshold` | 1,083/2,560 (42.30%) | 1.730859375 | 0.640625 | 0 |
| `proposed` | 1,410/2,560 (55.08%) | 1.34765625 | 1.000000 | 0 |
| `phase9_8_native_prefix_forecast_v1` | 1,514/2,560 (59.14%) | 1.40703125 | 1.090625 | 232 |

The naive policy aborted 920 attempts; each remains a failure. The prefix comparator's 232 selected intervention actions produced 108 recovery outcomes, 83 unnecessary-intervention outcomes, 37 ineffective-intervention outcomes, and 4 harm outcomes. Recovery is 108/1,150 scenarios whose continuation potential failed; unnecessary intervention is 83/1,410 continuation-passing scenarios; intervention harm is 4/232 selected intervention actions. The simulator's 4 harm outcomes are not actual harm caused by a deployed intervention.

The proposed policy selected `continue` in all 2,560 scenarios, exactly matching the no-intervention baseline. Its success and cost therefore tie that baseline in this pilot. The prefix comparator had a 4.06 percentage-point higher success rate than no-intervention but a higher mean realized cost by 0.059375 cost units. Its extra simulator information and unverified compute matching prevent a straightforward fairness or deployment interpretation. The naive threshold's 920 aborts correspond to lower success and higher mean cost than no-intervention.

## 5. Primary comparison and uncertainty

The pilot's preregistered primary estimand is the paired mean realized cost difference `proposed - no_intervention` over matched scenarios, aggregated through configuration-group means. Recalculation from `comparative_scored_runs.jsonl` gives **0.000000** across 2,560 scenarios and ten configuration groups. A 10,000-resample mechanism-stratified bootstrap with seed `20269810` gives a 95% percentile interval of **[0.000000, 0.000000]** (`experiments/phase9_8_pilot_runner.py`, `primary_paired_cost_estimand`).

The interval is degenerate because proposed and no-intervention chose the same action for every scenario, so every paired cost difference is zero. It is descriptive and is not a confirmatory significance test. There are only two pilot configuration groups per mechanism. The larger held-out design—eight groups per mechanism—was not evaluated.

The stored `comparative_metrics.json` contains the per-policy metrics but not the primary estimate or interval. Those were recomputed from the scored JSONL using the existing primary-estimand function and are reported here as recomputed quantities.

## 6. Interpretation

The central outcome is that the proposed learned policy never intervened and tied the no-intervention baseline. These records do not show that the classifier prevents failures or reduces unnecessary intervention. The prefix comparator's higher observed success alongside higher realized cost is a descriptive simulator trade-off under a privileged, simulator-assisted information set. The naive threshold's abort behavior harmed this pilot's measured success and cost; it does not establish how a different threshold or real critic would behave.

No ranking should be interpreted as evidence of general policy superiority. The model's output probabilities are not shown here to be empirically calibrated, and the input `failure_prob` is explicitly a synthetic score, not a calibrated probability. No live Gemini or other provider call contributed to these results.

## 7. Limitations and threats to validity

1. **Synthetic mechanisms and evidence.** Outcome equations and sensor-to-outcome relationships are illustrative simulator assumptions, not empirical estimates from real agents. Results cover five mechanism types and unseen configurations within them, not unseen mechanism families or real deployments.
2. **Pilot-only sample.** Ten configuration groups (two per mechanism) support only exploratory uncertainty summaries. The held-out partition remains unopened and unevaluated. No formal power claim is justified.
3. **Counterfactual outcomes.** Intervention, recovery, and harm are potential outcomes. The pilot does not execute an intervention, perform replanning, or test rollback behavior in a live trajectory.
4. **Comparator information and compute.** The native prefix comparator receives privileged simulator information, is distinct from the Phase 9.2 semantics, and its eight forecast units do not prove matched compute. Its result is not a deployability comparison.
5. **Incomplete saved potential-outcome detail.** The scored JSONL is sufficient to reproduce the published metrics, but the run directory does not contain a complete separate potential-outcomes JSONL for every scenario and action. Full evaluator-branch replay cannot be audited from that file alone.
6. **Approval-document wording.** The approved v4 lock and its hashes validate, but the v3 supersession and v4 technical addendum still display `DRAFT_NOT_APPROVED` in their text. The v3 supersession resolves the old comparator definition when paired with its approved lock; the stale header wording should be explained or corrected through an approved document process before submission.
7. **No full replay in the audit.** Summary metrics and the primary interval were recomputed from preserved records. The pilot itself was not rerun to test end-to-end deterministic replay.

## 8. Conclusion

The Phase 9.8 pilot's recorded metrics and primary cost interval are internally consistent and reproduce from the scored records. In this pilot, the learned proposed policy made no interventions and tied the no-intervention baseline. The simulator-assisted prefix comparator intervened in some counterfactual cases, but its additional information and higher mean cost must be reported alongside its success rate. The evidence supports a narrow conclusion about this frozen synthetic pilot only; it does not establish realistic critic calibration, intervention effectiveness, real-world policy superiority, or live LLM-agent performance.

## 9. Audit provenance

The audit checked the approved lock and its 45 files; decision-freeze manifest and records; scored runs; metrics JSON; and descriptive `pilot_test` selection. Exact output locations are listed in `reports/finalization/post_pilot_audit.md`. The audit recomputed policy metrics and the primary interval from stored scored rows. No experiment, model training/load, network call, or source/result modification was performed to prepare this report.
