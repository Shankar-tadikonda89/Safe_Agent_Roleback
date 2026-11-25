# Document Consistency Review

## Scope and evidence

This review compared the existing finalization documents—`reports/finalization/post_pilot_audit.md`, `reports/finalization/final_research_report.md`, and `reports/finalization/submission_checklist.md`—with the Phase 9.8 v3 supersession, v4 loader addendum, approved v4 lock, decision-freeze records, scored JSONL, and comparative metrics. The new submission-ready report and checklist are written against those audited records. No prior document, lock, model, result, or source file was changed.

## Findings

### Results and denominators

The existing finalization materials’ headline Phase 9.8 figures agree with the scored records. The audited comparison contains 2,560 scenarios and 10,240 decisions, with 2,560 decisions per comparative policy. Recalculation from scored JSONL agrees with the stored comparative metrics:

| Policy | Verifier success | Mean realized cost | Selected interventions |
|---|---:|---:|---:|
| `no_intervention` | 1,410 / 2,560 (55.08%) | 1.34765625 | 0 |
| `naive_threshold` | 1,083 / 2,560 (42.30%) | 1.730859375 | 0 |
| `proposed` | 1,410 / 2,560 (55.08%) | 1.34765625 | 0 |
| `phase9_8_native_prefix_forecast_v1` | 1,514 / 2,560 (59.14%) | 1.40703125 | 232 |

The proposed-minus-no-intervention mean cost difference is 0.0. Recalculation using 10,000 paired configuration-group bootstrap resamples gives the 95% interval [0.0, 0.0]. This estimate is recomputed from the scored records; the comparative metrics JSON does not itself store this estimate or interval. The equality in this pilot does not establish equivalence or generalization.

The naive policy’s 920 aborts remain failed outcomes. The prefix comparator selected intervention for 232 scenarios, but every comparative record indicates no confirmed application. Its recovery/unnecessary/ineffective/harm classifications are evaluator-side counterfactual simulator outcomes, not observed effects of actions actually applied to an agent.

`pilot_test` is stored separately, selected using pilot outcomes, and marked ineligible for comparative metrics. It chose continue in this run; its descriptive selection costs are not included in the four-policy comparison.

### Incomplete potential-outcomes records

The freeze and scored JSONL provide matched decisions, provenance, realized outcome fields, and enough recorded data to reproduce the reported policy metrics and primary paired comparison. The run directory does not provide a complete per-scenario log of both potential branches, including every unselected branch. Consequently, a reader cannot independently reconstruct all branch-level counterfactual outcomes from the preserved scored records. The submission report states this boundary and does not invent missing records.

### Protocol and comparator wording

The original protocol/preregistration contain the earlier `prefix_branching_approximation` identity and shared-pass-probability formulation. The v3 supersession explicitly replaces those conflicting comparator terms, and the v4 adoption addendum specifies the sole Phase 9.8 comparator, action-specific pass probabilities and costs, strict lower-cost selection, tie-to-continue, and four forecasts per action. The approved v4 lock binds the later approval package. The new research report uses the superseding Phase 9.8 definitions and does not describe the comparator as an exact Phase 9.2 port.

The comparator uses privileged simulator information unavailable to the proposed policy. It also has eight forecasts per decision, with no proven compute matching. These are stated as limitations, not hidden as a fair deployment comparison.

### Draft-status inconsistency requiring manual resolution

The v3 supersession and v4 addendum retain `DRAFT_NOT_APPROVED` status wording, including text that describes a lock as draft, while the corresponding v3 and v4 lock records are approved. This may reflect historical drafting status, but the documents themselves do not provide a self-contained approval-status clarification sufficient to resolve that appearance for an external reader.

**Required manual action:** preserve those historical, lock-bound documents and both locks. Before submission, have the responsible human approver or institution decide whether a separately versioned document-control cover note is sufficient. If so, it should identify the relevant approved lock versions, exact raw hashes, and human approval record, and explain that the bound addenda retain their historical draft-status labels. If institutional policy instead requires the text itself to say approved, create separately versioned corrections and obtain a separately reviewed/reapproved lock package; do not edit the bound files or mutate the approved lock in place.

### README distinction

The README has a separate “Phase 9.2” section reporting 7,680 scenarios and different policy results. Those are not the Phase 9.8 pilot figures. The new report identifies the Phase 9.8 run and uses its own lock, freeze, and scored-log provenance. The README was not modified in this document-only finalization, so its separate historical section still needs editorial review to ensure it is not mistaken for the final Phase 9.8 result.

## Agreement of the new submission documents

The new research report and checklist agree on the four comparative policies, denominators, success rates, mean costs, intervention counts, primary paired cost estimate and interval, `pilot_test` exclusion, one-decision counterfactual semantics, missing full branch log, and synthetic/privileged-comparator limitations. The checklist points to this review for the draft-status discrepancy and README distinction. No new metric, significance claim, citation, live-provider result, or real-world validation claim has been introduced.

## Conclusion

The new submission documents are numerically consistent with the audited Phase 9.8 scored records. The historical draft-status wording and README distinction remain editorial/document-control items for human review. This review does not authorize changes to bound documents or locks and does not claim the package is ready for unconditional submission.
