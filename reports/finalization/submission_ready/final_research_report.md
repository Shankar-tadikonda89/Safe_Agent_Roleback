# Safe Intervention Without Rollback
## Final Phase 9.8 Synthetic Pilot Report

### Abstract

This report evaluates a learned four-class intervention policy in a frozen offline synthetic pilot. Four policies were compared on 2,560 matched scenarios from ten configuration groups across five simulator mechanisms. The proposed policy selected `continue` in every scenario, tying `no_intervention` at 55.08% verifier success and 1.34765625 mean synthetic cost units. The paired proposed-minus-no-intervention cost difference was 0.0; its 95% mechanism-stratified configuration-group bootstrap interval was [0.0, 0.0] over 10,000 resamples. The `naive_threshold` policy aborted 920 scenarios and had lower success and higher mean cost. A simulator-assisted prefix comparator selected intervention in 232 scenarios and had the highest observed success rate, but also a higher mean cost than `no_intervention`; it used privileged simulator information and has no demonstrated compute matching. All results are counterfactual simulator results. No intervention was confirmed as applied, no held-out partition was evaluated, and this pilot establishes neither real-world effectiveness nor statistical significance.

## Motivation and contribution

The capstone research question is whether a lightweight, single-pass decision policy can use inexpensive trajectory observations to decide when to intervene, reducing unnecessary disruption without counterfactual re-execution at decision time. The research contribution is the implementation and evaluation of a cost-sensitive policy interface, a four-class potential-outcome objective, and a matched synthetic comparison framework. The pilot’s result is a negative or null behavioral finding for the proposed policy: it did not intervene and tied the no-intervention baseline. It does not show that intervention prevents failures in real agent systems.

This study used local deterministic simulation. It did not run live Planner/Coder/Verifier agents or call Gemini. “Success,” “recovery,” “unnecessary intervention,” and “harm” below refer to verifier outcomes in the synthetic simulator.

## Decision features and proposed policy

The proposed classifier consumes four features in this fixed order: `failure_prob`, `reversibility_proxy`, `trajectory_position`, and `disruption_history`.

- **`failure_prob`** is a legacy feature name for an explicitly uncalibrated synthetic risk score. It combines visible anomaly score, anomaly alert, capped error count, and capped warning count with equal 0.25 weights. Its stated target is rejection at the next verifier checkpoint, or final verification if no intermediate checkpoint exists. It is not an empirically calibrated probability.
- **`reversibility_proxy`** maps pending action metadata as `read_only=0.95`, `reversible_write=0.65`, and `irreversible_write=0.20`. It is a proxy, not a measured probability of reversibility.
- **`trajectory_position`** is the zero-based plan position divided by `planned_step_count - 1`; a one-step plan maps to 0.
- **`disruption_history`** is `min(n/3, 1)`, where `n` is the count of prior `intervention_applied` events in the complete append-only ledger prefix. Events at or after the current decision are excluded.

The fitted model is `StandardScaler` plus multinomial scikit-learn `LogisticRegression`. `C=1.0` was selected from `{0.1, 1, 10}` by validation multiclass log loss under the frozen lower-`C` tie rule. No post-hoc probability calibrator was used. The classifier was trained by maximum likelihood; cost sensitivity is applied at inference through the expected-cost rule.

### Cost matrix and decision rule

| Potential-outcome class | Continue cost | Intervene cost |
|---|---:|---:|
| `recovery` | 3 | 2 |
| `unnecessary_intervention` | 0 | 2 |
| `ineffective_intervention` | 3 | 5 |
| `intervention_harm` | 0 | 5 |

For class probabilities `(q_recovery, q_unnecessary, q_ineffective, q_harm)`:

```text
E_continue  = 3 * (q_recovery + q_ineffective)
E_intervene = 2 + 3 * (q_ineffective + q_harm)
```

The proposed policy selects the lower expected cost; ties continue. It makes one prediction per decision and does not branch or roll back at inference.

## Experimental setup

The frozen synthetic design contains 120 configuration groups over five mechanisms: transient tool failure, stale/contradictory evidence, irreversible action, late trajectory failure, and replanning disruption. Each group contains 256 scenario seeds. The pilot used ten groups (two per mechanism), 2,560 scenarios, and root seed `20269804`. The remaining 40 held-out groups were not generated or evaluated. The design is exploratory; configuration groups, rather than individual scenarios, are the uncertainty units.

All four comparative policies were matched on scenario, seed, configuration group, and initial-state hash. A complete 10,240-decision set was frozen and reread before evaluator outcomes were joined. Policies were:

1. **`no_intervention`** — always continue.
2. **`naive_threshold`** — abort when the uncalibrated synthetic risk score is strictly greater than 0.5. Abort is a failure, not an intervention.
3. **`proposed`** — use the saved four-class classifier and cost matrix above.
4. **`phase9_8_native_prefix_forecast_v1`** — take four simulator-assisted forecasts per action (eight per scenario), compute separate action-specific pass estimates, and choose the lower cost:

   ```text
   C_continue  = 3 * (1 - p_pass_continue)
   C_intervene = 2 + 3 * (1 - p_pass_intervene)
   ```

   Ties continue. This comparator is a Phase 9.8-native simulator-assisted method, not an exact Phase 9.2 port. It receives privileged simulator susceptibility/configuration information unavailable to `proposed`; equal compute and deployability are unproven.

The outcome-selected `pilot_test` policy chose one global action using pilot outcomes. It is reported separately and excluded from comparative metrics. Its descriptive record selected `continue` (continuation mean cost 1.34765625; intervention mean cost 2.655078125); this is in-sample selection, not a fifth baseline result.

### Counterfactual execution semantics

This is a one-decision counterfactual simulator, not an executed multi-step intervention and replan workflow. The evaluator computes potential verifier outcomes after decisions are frozen. No tool operation, repair, plan mutation, or replan is executed. Scored rows record `confirmed_application=false` and `intervention_applied=null`. Therefore the comparator’s 232 selected interventions are counterfactual action choices; they were not interventions actually applied. Recovery and harm labels describe simulated potential outcomes, not observed operational effects.

## Verified results

Metrics below were recalculated from the existing scored JSONL and exactly match `comparative_metrics.json`. Success is verifier-passing attempts divided by 2,560. Mean realized cost and mean steps are per attempt. “Selected interventions” counts policy decisions to intervene, not confirmed application.

| Policy | Verifier success | Mean realized cost | Mean steps | Selected interventions |
|---|---:|---:|---:|---:|
| `no_intervention` | 1,410/2,560 (55.08%) | 1.34765625 | 1.000000 | 0 |
| `naive_threshold` | 1,083/2,560 (42.30%) | 1.730859375 | 0.640625 | 0 |
| `proposed` | 1,410/2,560 (55.08%) | 1.34765625 | 1.000000 | 0 |
| `phase9_8_native_prefix_forecast_v1` | 1,514/2,560 (59.14%) | 1.40703125 | 1.090625 | 232 |

The naive policy aborted 920 scenarios; every abort remains a failure. For the prefix comparator, the 232 selected intervention outcomes comprised 108 recoveries, 83 unnecessary interventions, 37 ineffective interventions, and 4 intervention-harm outcomes. The recorded denominators are 1,150 continuation failures for recovery, 1,410 continuation passes for unnecessary intervention, and 232 selected actions for harm.

## Primary comparison and uncertainty

The primary paired estimate is the mean realized cost difference `proposed - no_intervention` over the 2,560 matched pilot scenarios, aggregated through configuration-group means. Recomputed from the scored records, the estimate is **0.0** across ten groups. The paired mechanism-stratified 95% percentile interval is **[0.0, 0.0]**, using **10,000 configuration-group bootstrap resamples** and seed **`20269810`**.

The interval is degenerate because the proposed policy continued in every scenario and therefore had the same paired cost as `no_intervention` in every scenario. It is descriptive, not a significance test or proof of equivalence. The pilot has only two configuration groups per mechanism; the larger held-out design has eight per mechanism and was not evaluated.

The primary estimate and interval do not appear in the stored metrics JSON. They were recomputed using the existing `primary_paired_cost_estimand` function from `comparative_scored_runs.jsonl` and are reported here as derived results.

## Interpretation

The proposed policy did not demonstrate useful intervention behavior in this pilot: it made no interventions and tied the no-intervention baseline. The prefix comparator’s observed success rate was higher than no-intervention, but its mean cost was also higher by 0.059375 synthetic cost units per scenario. Its privileged simulator information and unproven compute matching prevent a fair deployment or superiority interpretation. The naive threshold’s aborts were failures and coincided with lower success and higher cost in this pilot. No result supports a claim of real-world effectiveness.

## Limitations and threats to validity

1. **Synthetic evidence:** simulator equations, features, and sensor-to-outcome relationships are illustrative assumptions, not empirically validated observations.
2. **Limited pilot configurations:** ten groups, two per mechanism, provide a small basis for group-level uncertainty. There is no formal power claim; held-out evaluation remains undone.
3. **Counterfactual rather than applied actions:** recovery, unnecessary intervention, ineffectiveness, and harm are simulated potential outcomes. The study does not establish what real interventions would do and does not test live rollback behavior.
4. **Information and compute asymmetry:** the prefix comparator has privileged simulator access. Eight forecast units do not establish matched compute, and the comparator is not the original Phase 9.2 implementation.
5. **Incomplete potential-outcome archive:** the run directory has no complete evaluator-potential-outcomes JSONL. The scored rows suffice to reproduce reported summary metrics, but unselected intervention potentials are not all independently reconstructable from the saved run records. The descriptive `pilot_test` record preserves aggregate selection costs, not per-scenario inputs.
6. **Primary metric storage:** `comparative_metrics.json` omits the primary paired estimate and bootstrap interval; this report identifies them as recomputed from scored records.
7. **Document-status inconsistency:** the v3 supersession and v4 technical addendum still display `DRAFT_NOT_APPROVED`, although the corresponding lock versions are approved. The lock was not changed to repair the prose; manual document-control clarification remains necessary.
8. **No repeat-run audit:** this report recomputed metrics from preserved records but did not rerun the pilot to test end-to-end deterministic replay.
9. **No live model evidence:** no Gemini/provider calls were made; synthetic classifier output is not evidence about deployed LLM agents.

## Conclusion

The integrity and recorded summary results for the Phase 9.8 pilot are internally consistent. In the recorded synthetic sample, the proposed classifier policy made no interventions and tied the no-intervention baseline. The simulator-assisted comparator selected some counterfactual interventions but had privileged information and a higher mean cost than no-intervention. The pilot supports only a narrow descriptive conclusion about the frozen synthetic scenarios; it establishes neither intervention effectiveness nor real-world policy superiority.

### Provenance

The approved v4 lock has raw SHA-256 `893a212550183db3315858cc9cd46210674c548da664f6f5a1d2007d26fabd72` and canonical SHA-256 `172663ff409325929e4d3b804fa33cab407bb4fcf9a41ed935ae340f0b513dff`; all 45 bound hashes verified. The decision freeze is `freeze-3d4f75d0ae811b9a7342`; the scored run is `run-3d4f75d0ae811b9a7342`. The record files, lock, source functions, and versioned supersession documents are identified in the accompanying consistency review. No experiment was rerun for this report.
