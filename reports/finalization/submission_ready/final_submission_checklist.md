# Final Submission Checklist

This checklist covers the approved Phase 9.8 pilot and its submission documents. It does not replace institutional formatting, citation, or authorship requirements.

## Completed and verified

- [x] Read and cross-checked the existing post-pilot audit, final research report, and submission checklist.
- [x] Verified the approved v4 pilot lock status and its recorded raw and canonical SHA-256 digests; all 45 bound-file hashes passed in the read-only integrity audit.
- [x] Verified the decision-freeze records and scored records: 2,560 matched scenarios and 10,240 comparative decisions, 2,560 for each of four policies.
- [x] Recomputed comparative metrics from the scored JSONL records and confirmed they equal `results/phase9_8_native_pilot_v4/runs/run-3d4f75d0ae811b9a7342/comparative_metrics.json`.
- [x] Recomputed the proposed-minus-no-intervention mean realized cost difference as 0.0 and its paired configuration-group bootstrap 95% interval as [0.0, 0.0], using 10,000 resamples over 10 configuration groups and the preregistered bootstrap seed.
- [x] Verified the four comparative policy summaries: no intervention 55.08% success, mean cost 1.34765625, 0 interventions; naive threshold 42.30%, mean cost 1.730859375, 0 interventions and 920 aborts; proposed 55.08%, mean cost 1.34765625, 0 interventions; native prefix forecast 59.14%, mean cost 1.40703125, 232 selected counterfactual interventions.
- [x] Verified that aborts are failures unless the verifier passes; the naive-threshold aborts are not reported as successful interventions.
- [x] Verified all scored runs report no confirmed intervention application. The prefix comparator’s 232 selections and associated outcome categories are simulator counterfactuals, not interventions actually applied.
- [x] Verified `pilot_test` is recorded separately as descriptive outcome-selected information and is excluded from comparative metrics.
- [x] Documented the synthetic scenario/task limitations, fault-derived feature concerns where relevant, privileged simulator information available to the prefix comparator, and absence of live Gemini/provider calls.
- [x] Created a new research report, checklist, and document-consistency review under `reports/finalization/submission_ready/`. Historical audit documents and artifacts remain in place.

## Documented limitations

- The pilot is a one-decision synthetic counterfactual simulator, not an executed multi-step agent intervention study. A selected action does not establish that repair/replanning was actually applied.
- The evaluation covers 10 pilot configuration groups and is exploratory; the held-out partition was not generated or evaluated in this pilot. The reported interval is a degenerate paired estimate because proposed and no-intervention decisions/outcomes coincide in these records; it is not evidence of equivalence or generalization.
- The saved scored records support the reported comparative metrics and primary paired estimate, but the run folder does not contain a complete potential-outcomes record for every unselected branch. Those missing branch records cannot be independently reconstructed from the available artifacts.
- The comparative metrics JSON contains the policy summary but not the primary paired estimate and confidence interval; those values are explicitly reported as recomputed from the scored JSONL, not copied from that JSON file.
- The native prefix comparator is Phase 9.8-native and simulator-assisted. It has privileged information and its compute matching and deployability have not been established; it is not an exact Phase 9.2 port.
- These synthetic results do not establish real-world effectiveness, live-LLM performance, classifier calibration in deployment, or statistical significance.
- The project README’s Phase 9.2 section describes a separate experiment with different scenario counts and results. It must not be presented as the Phase 9.8 pilot.

## Manual review required before submission

- [ ] Resolve the document-status wording mismatch described in `document_consistency_review.md`. The v3 supersession and v4 addendum retain `DRAFT_NOT_APPROVED` wording while their corresponding locks are approved. Preserve the approved locks and bound historical files. Add an institutionally acceptable, separately versioned document-control clarification referencing the exact lock digests and human approval, or obtain review on whether the historical draft labels are acceptable as-is. Do not edit bound documents in place or change a lock without a separately reviewed and authorized process.
- [ ] Have the capstone supervisor/research reviewer confirm that the one-decision counterfactual scope, comparator information asymmetry, and incomplete potential-outcomes record are described adequately for the submission.
- [ ] Reconcile the README’s separate Phase 9.2 summary with the final submission narrative so readers can distinguish that experiment from the Phase 9.8 pilot. The README was not modified as part of this document-only task.
- [ ] Add or verify scholarly references required by the institution. No external citations were invented for this finalization; the report relies on project artifacts and records.
- [ ] Complete institution-specific formatting, authorship, and submission checks.

## Readiness

The new documents are internally aligned with the audited Phase 9.8 records. The submission package is **not fully ready for submission until the manual document-status clarification/review and institutional checks above are completed**.
