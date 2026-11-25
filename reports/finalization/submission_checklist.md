# Phase 9.8 Submission Checklist

## Verified and ready to include

- [x] Include the final read-only audit: `reports/finalization/post_pilot_audit.md`.
- [x] Include the research report: `reports/finalization/final_research_report.md`.
- [x] Cite the approved v4 lock status and hashes: raw `893a212550183db3315858cc9cd46210674c548da664f6f5a1d2007d26fabd72`; canonical `172663ff409325929e4d3b804fa33cab407bb4fcf9a41ed935ae340f0b513dff`.
- [x] Cite the preserved decision freeze `freeze-3d4f75d0ae811b9a7342` and scored run `run-3d4f75d0ae811b9a7342`.
- [x] Report exactly four comparative policies and 2,560 matched scenarios / 10,240 decisions; report `pilot_test` separately as outcome-selected descriptive data.
- [x] State that the proposed policy selected no interventions and tied the no-intervention baseline on this pilot.
- [x] Report the primary estimate and [0, 0] interval as recomputed from scored records, with 10,000 group-bootstrap resamples and seed `20269810`.
- [x] Label all outcomes as synthetic and counterfactual. Do not describe any selected intervention as an action actually applied.

## Required disclosures in the submitted paper/slides

- [ ] Identify the interval as a pilot-partition result with ten groups (two per mechanism), not as held-out evaluation. The 40-group held-out partition was not evaluated.
- [ ] Explain that the degenerate primary interval follows from identical proposed and no-intervention decisions, not from demonstrated equivalence or policy success. Make no formal significance or power claim.
- [ ] Explain that the native prefix comparator receives privileged simulator information, is not an exact Phase 9.2 port, and has no demonstrated compute matching or deployment readiness.
- [ ] State that the prefix comparator selected 232 counterfactual interventions; 108 recovery, 83 unnecessary, 37 ineffective, and 4 harm outcomes were simulator outcomes, not observed operational effects.
- [ ] State that all selected interventions have `confirmed_application=false`; the pilot did not execute tool actions, repairs, replanning, or rollback behavior.
- [ ] Distinguish the uncalibrated synthetic `failure_prob` score from a calibrated probability and from the proposed model's four-class probabilities.
- [ ] Do not claim live Gemini/provider experiments or real-world LLM-agent effectiveness; this pilot made no provider calls.

## Evidence and documentation items to resolve

- [ ] Preserve the separate audit note that `comparative_metrics.json` does not store the primary paired estimate or interval; those figures were recomputed from `comparative_scored_runs.jsonl` using the existing primary-estimand function.
- [ ] Disclose that no complete per-scenario evaluator potential-outcomes JSONL was saved. The published metric summaries reproduce from the scored JSONL, but all unselected counterfactual branches cannot be reconstructed from the preserved run directory alone.
- [ ] Explain the protocol version chain: the original protocol and pilot preregistration retain a superseded prefix-comparator definition. Include the v3 supersession and v4 wrapper-loader addendum with the approved lock evidence. Their document headers still say `DRAFT_NOT_APPROVED`; resolve or explicitly explain that stale wording through the appropriate approval/document-control process before submission.
- [ ] Confirm the README and presentation describe this Phase 9.8 pilot without substituting historical Phase 9.2/9.7 results for it. No README change was made during finalization.
- [ ] Add and verify institution-required scholarly references independently. This report invents no external citations; source and protocol references are project-local.

## Scope of what was verified

The approved lock's hashes, freeze-record and manifest integrity, matched policy coverage, run counts, per-policy metrics, and the primary bootstrap interval were verified from preserved files. The summary metrics and interval are reproducible from the scored run records. The pilot was not rerun; end-to-end repeated-run identity was not newly tested. No source, frozen artifact, lock, decision freeze, scored run, or existing result was changed to prepare the finalization reports.
