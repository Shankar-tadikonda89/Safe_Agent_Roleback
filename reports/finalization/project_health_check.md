# Full Project Health Check

**Project:** Safe Intervention Without Rollback  
**Audit scope:** Read-only project review with isolated tests and demonstrations  
**Overall verdict:** **B. Ready after minor fixes** for an **offline synthetic demonstration**. The live Gemini path and Streamlit UI are not verified as runnable services, and the full 202-test suite was not run because it includes model training and artifact-loading cases prohibited by the audit constraints.

## Executive summary

The deterministic offline CLI works in the current Python environment. I ran one proposed-policy task against the existing Phase 4 artifact and observed a verifier pass. I also exercised the intervention/replan path with a temporary fake policy; the current code retained the existing code prefix and passed verification. The Phase 9.8 fitted model loaded through its approved v4 lock gate and produced a valid single decision on an explicit feature row. These checks did not run the pilot, create scenarios for an experiment, or call a provider.

The approved Phase 9.8 pilot records remain internally consistent: the lock and all 45 bound files verify, the decision record digest matches its freeze manifest, and recalculated metrics from the 10,240 scored rows equal the stored metrics. The primary proposed-minus-no-intervention mean cost difference is 0.0 with a 95% interval of [0.0, 0.0] over 10,000 configuration-group bootstrap resamples. This result reflects identical actions/outcomes in this synthetic pilot; it does not establish equivalence, effectiveness, or generalization.

The most important follow-up items are maintenance and presentation issues: a few legacy lock tests still expect historical draft locks, README status rows are stale relative to later phases, and the repository has no dependency manifest. A strict probability-type check in the older CLI feature extractor accepts booleans because Python treats `bool` as a subclass of `int`. The optional dashboard cannot be launched in this environment because Streamlit is not installed. No provider request was made; provider connectivity remains unverified.

## 1. Architecture and execution flow

### Major component map

```text
Offline CLI (`run_demo.py`)
  main -> run -> Planner.plan -> decision-time feature extraction
       -> no_intervention / naive_threshold / Phase 4 ProposedPolicy
       -> optional corrective guidance + Planner.replan
       -> Coder.implement -> Verifier.verify -> JSONL trace + summary

Live CLI (`run_demo.py --mode live`)
  main -> agents.live_gemini.run_live -> project-local config loader
       -> locked GeminiJsonClient -> LivePlanner / LiveCritic / LiveCoder /
          LiveVerifier -> deterministic local oracle -> separate live JSONL

Phase 9.5/9.7 foundations
  immutable observation + provenance -> append-only event history
       -> simulator/runner allowlist -> policy decision before evaluator join

Phase 9.8 pilot (already completed; not rerun in this audit)
  v4 approved-lock validation -> wrapped Phase 9.8 model extraction
       -> frozen pilot design + factories -> matched policy decisions
       -> persisted/reread decision freeze -> evaluator outcomes -> scoring
       -> comparative metrics + separate descriptive `pilot_test` record

Optional dashboard
  dashboard/app.py -> dashboard/helpers.py -> existing offline runner and
  existing metrics reader; it does not implement a second policy engine.
```

The legacy CLI is an eight-task arithmetic simulator. The Phase 9.8 pilot is a separately versioned synthetic, one-decision counterfactual study. These paths should not be presented as the same experiment. In Phase 9.8, the prefix comparator's selected actions are counterfactual choices; the pilot does not execute a repair or replan.

### Component health

| Component | Status | Evidence and limits |
|---|---|---|
| Offline task initialization and deterministic scenario | **PASS** | `run_demo.py` defines the eight tasks and seeded fault scenario. The offline CLI completed task 02, seed 42, using the saved proposed model. |
| Planner, Coder, and Verifier | **PASS** | Source is deterministic and constrained; the coder parses arithmetic expressions under a node allowlist and the verifier checks the expected answer. Unit tests cover the prompt boundary and verifier behavior. |
| Decision-time feature extraction | **WARNING** | The current arithmetic CLI deliberately returns the neutral 0.5 risk prior. It does not use the fault flag. However, `extract_features` accepts `True` as a numeric risk value; see confirmed finding below. |
| Phase 4 proposed policy | **PASS** | Existing `models/proposed_policy.joblib` loaded through the legacy policy loader during the offline CLI demonstration and produced a decision. This is the older binary helpfulness model and is distinct from Phase 9.8. |
| Phase 9.5 observation/event foundations | **PASS** | Focused tests passed for provenance, immutable observations, event prefixes, lifecycle rules, and no rollback. |
| Phase 9.7 runner and cost rule | **PASS** | Focused integration tests passed for event ordering, feature/evaluator separation, cost arithmetic, and single-call inference contracts. |
| Phase 9.8 approved model adapter | **PASS** | The actual saved Phase 9.8 artifact loaded via `load_native_pilot_adapter_v4` after the approved lock/hash gates; a single four-feature decision returned `continue`. This was not a scenario or policy comparison. |
| Phase 9.8 decision freeze and results | **PASS** | Current lock raw/canonical hashes and all 45 bound hashes passed. Freeze record digest passed. 10,240 scored rows recalculate exactly to the stored comparative metrics. |
| Phase 9.8 counterfactual/intervention semantics | **PASS** | The frozen records say no intervention was confirmed as applied. `pilot_test` is separate and absent from comparative metrics. |
| Live Gemini runtime | **NOT TESTED** | Fake SDK tests passed. No API/provider request was made in this audit. Historical smoke-test output was a sanitized failure, so successful provider connectivity is not established. |
| Streamlit dashboard runtime | **NOT TESTED** | `streamlit` is not installed. Dashboard helper tests and source syntax check passed, but the UI process was not launched. |
| Dependency installation/reproducibility | **WARNING** | Python is 3.13.16 and required core packages are present, but there is no root `requirements.txt` or `pyproject.toml` to reproduce versions. No packages were installed. |
| Full regression suite | **WARNING** | 202 unittest cases are discoverable. Safe focused tests passed, but the full suite was not run because it includes training and model artifact tests; an initial safe batch exposed four stale legacy-lock test failures. |

## 2. Environment and dependencies

- Python: **3.13.16**. The master spec requires Python 3.11 or newer, so the interpreter meets the stated minimum.
- Installed versions observed through package metadata: scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, Google GenAI 2.31.0, and joblib 1.6.0.
- `streamlit` and `matplotlib` are not installed. The optional dashboard requires Streamlit. The evaluation chart path uses standard-library SVG generation, so missing matplotlib did not block the tested core workflows.
- The project root has neither `requirements.txt` nor `pyproject.toml`. The master specification listed a requirements file, so setup is not reproducible from a declared dependency set. This audit did not install or upgrade dependencies.
- `.gitignore` excludes `.env`, `models/`, `results/`, and `logs/`; `.env.example` contains placeholders and the locked model name. I inspected neither `.env` values nor credentials. `agents.live_gemini.load_project_gemini_config` reads the project-local `.env` with `dotenv_values`, enforces `gemini-3.8-flash`, and the fake-SDK tests verify request timeout/retry configuration without contacting the provider.
- The v4 model and metadata files are present. The Phase 4 default model is also present. Both were only loaded through their existing expected application paths for the safe demonstration; no model was trained or saved.

## 3. Static code quality and interface findings

### Confirmed issue: booleans pass the older CLI feature validator

At `run_demo.py:146`, `failure_prob` is checked with `isinstance(failure_prob, (int, float))`. Since `bool` is a subclass of `int`, `extract_features(True, "code_assignment", 0, 1)` currently returns `failure_prob=1.0`. I verified this with a read-only in-memory call. Normal CLI feature generation currently supplies the fixed float `0.5`, so this did not affect the demonstration. The Phase 9.8 adapter has explicit boolean rejection; the older extractor should use a strict numeric check if it remains a supported public interface.

### Potential safety limitation: legacy joblib loader

`policy/proposed.py:44-63` loads the Phase 4 joblib artifact directly and then validates its structure. Joblib/pickle deserialization occurs before structural validation. This is acceptable for the project's locally generated, trusted artifact, but the legacy path should not load arbitrary user-supplied model files. The Phase 9.8 pilot path uses the separate approved v4 hash gate before deserialization (`experiments/phase9_8_native_pilot_loader_v4.py:236` onward).

### Imports and syntax

An AST parse succeeded for all **62 Python files**. The selected test imports exercised the core CLI, policy, simulator, dashboard helper, and live-wrapper modules without import errors. `dashboard/app.py` itself cannot be runtime-imported here because it depends on missing Streamlit; the existing dashboard test checked its syntax without importing Streamlit.

## 4. Functional verification

### Offline CLI

Command:

```powershell
python -B run_demo.py --task task_02 --seed 42 --mode offline --policy proposed --log <unique-file-under-%TEMP%>
```

Actual result: clean scenario; policy continued; coder produced `answer = 42`; verifier passed; process exit code 0. The trace was written under the operating system temporary directory, not under project `logs/` or `results/`.

### Actual intervention path, using an injected test policy

A small temporary Python check patched only the in-process policy loader to return a fixed `intervene` decision. It ran task 01/seed 42 through the normal `run_demo.run` path and confirmed `intervention_count=1`, `planner_replanned=True`, preserved the prefix (`code_before=''` on the first step), and verified the corrected answer. The log was in a temporary directory. This verifies the mechanics path, not the learned classifier's ability to choose a useful intervention.

### Phase 9.8 adapter

The v4 loader checked the approved lock and its bindings before loading the saved wrapped model. A single direct adapter prediction using a valid four-feature mapping returned `continue` with four class probabilities. No scenarios, counterfactual branches, outcomes, pilot metrics, or policy comparison were generated.

## 5. Decision policy and cost verification

For the Phase 9.8 four-class model, the order is recovery, unnecessary intervention, ineffective intervention, and intervention harm. The frozen cost matrix is continue `(3, 0, 3, 0)` and intervene `(2, 2, 5, 5)`. `experiments/phase9_7_runner.py:105-132` computes the expected costs and continues on ties; `experiments/phase9_8_adapter.py:166-196` validates the ordered features and probability vector, calls the classifier once, and returns the selected action.

The older Phase 4 CLI uses a separate two-class helpfulness matrix in `policy/proposed.py`. Its formula is `continue=3p` and `intervene=2-p`, with strict lower-cost selection. These policy versions must not be described as having one shared cost matrix.

The Phase 9.8 tests passed for feature/class order, invalid probability rejection, one prediction per decision, and tie-to-continue behavior. The risk feature is not a validated real-world critic probability: in the arithmetic CLI it is a neutral 0.5 prior; in Phase 9.7 it is an uncalibrated simulator score; Phase 9.8 uses the frozen synthetic schema/model. The Phase 9.8 native prefix comparator has privileged simulator information and eight forecast computations per decision, so compute matching and deployability are not established.

## 6. No-rollback behavior

The executable offline CLI appends corrective guidance to the current instruction, replans at the current step, and passes accumulated code forward to the coder (`run_demo.py:275-313`). The injected-intervention check verified that behavior for one task. Phase 9.5/9.7 tests passed for append-only event history, event ordering, and rejection of rollback in the irreversible-action fixture.

Phase 9.8 has different semantics: it is a one-decision counterfactual simulator. The audited scored rows record `confirmed_application=false` and `intervention_applied=null`. Therefore the prefix comparator's 232 selected actions and simulated recovery/harm classifications are not actual applied interventions.

## 7. Phase 9.8 integrity and verified results

Read-only integrity command logic recomputed the current approved v4 lock's raw and canonical SHA-256 hashes, compared all 45 bound files, checked the freeze record hash, and recalculated metrics from the scored JSONL. Results:

- v4 lock status: `APPROVED`; raw SHA-256 `893a212550183db3315858cc9cd46210674c548da664f6f5a1d2007d26fabd72`; canonical SHA-256 `172663ff409325929e4d3b804fa33cab407bb4fcf9a41ed935ae340f0b513dff`.
- **45/45 bound-file hashes match.** Decision records SHA-256 matches the freeze manifest.
- **10,240** scored records: 2,560 per comparative policy, across 2,560 matched scenarios.
- Recomputed `calculate_pilot_metrics(scored_rows)` equals the stored `comparative_metrics.json` exactly.
- `pilot_test` does not appear in the comparative metrics and is marked descriptive-only in its separate record.

| Policy | Verifier success | Mean realized cost | Selected interventions |
|---|---:|---:|---:|
| `no_intervention` | 1,410/2,560 (55.08%) | 1.34765625 | 0 |
| `naive_threshold` | 1,083/2,560 (42.30%) | 1.730859375 | 0; 920 aborts remain failures |
| `proposed` | 1,410/2,560 (55.08%) | 1.34765625 | 0 |
| `phase9_8_native_prefix_forecast_v1` | 1,514/2,560 (59.14%) | 1.40703125 | 232 counterfactual selections |

The primary proposed-minus-no-intervention mean cost difference is **0.0**, with a **95% interval [0.0, 0.0]**, from 10,000 paired mechanism-stratified configuration-group bootstrap resamples. It is a recomputation from existing scored rows; the stored comparative metrics JSON does not include the primary interval. The proposed model never intervened and tied the no-intervention baseline. No policy superiority or statistical-significance claim is supported.

The run directory contains no complete evaluator potential-outcomes JSONL. Existing scored rows support the stated aggregate metrics and primary difference, but cannot independently reconstruct every unselected branch for each scenario or every per-scenario input used by `pilot_test`. Missing records were not fabricated.

## 8. Regression tests

`python -B -c "import unittest; s=unittest.defaultTestLoader.discover('tests'); print('unittest_discovered_test_cases='+str(s.countTestCases()))"` discovered **202 test cases** without running them.

The complete suite was **not run** because it includes tests that train models or load serialized artifacts in Phase 1, Phase 8.3, Phase 9.2 evaluation, Phase 9 cost-sensitive training, and Phase 9.8 training. That would violate the no-retraining constraint. No tests that run the frozen pilot or full experiment were executed.

Safe focused tests executed:

- Phase 9.5 foundations, Phase 9.7 integration, Phase 9.8 decision freeze, adapter, native loader contracts, pilot runner, prefix forecast, pilot factory boundary, Phase 9.2 simulator, dashboard helpers, fake Gemini live behavior, and smoke helper: **123 passed** in the clean focused invocation.
- Phase 1 and Phase 3 core behavior: **12 passed**.
- Total successful focused executions: **135 tests passed**.

The first broad safe batch ran 127 cases and produced **123 passes, 3 errors, and 1 failure** in four historical lock tests. These are confirmed stale test expectations, not current v4 runtime failures:

1. `test_valid_test_fixture_reaches_only_injected_fake_loader` (v2) expects the historical v2 bound-file hashes to match current files; they do not.
2. `test_actual_draft_lock_fails_before_fake_deserializer` (v3) assumes the v3 lock remains draft; it is approved.
3. `test_draft_lock_stops_before_scenario_or_evaluator_factories` exercises the old v3 orchestrator against the now-approved v3 lock; it no longer stops at the draft gate.
4. `test_real_draft_lock_rejects_before_injected_artifact_loader` (v4) assumes the v4 lock remains draft; it is approved, so its injected fake loader is called and the assertion fails.

No real model deserialization occurred in those failing tests; the loaders were injected fakes. The unaffected fake-loader tests passed when rerun separately. These historical tests should be updated to use versioned temporary draft fixtures or current approval expectations before claiming the entire suite is green.

Other executed checks: AST syntax parse passed for 62 Python files; lock/freeze/result integrity passed; CLI demo exited 0; approved v4 one-row inference exited 0; and temporary fake-policy intervention mechanics passed. One audit-only one-line command initially requested the wrong result key (`decision` rather than the adapter's `selected_action`), causing a diagnostic `KeyError` after the loader and prediction completed; the corrected read-only check passed.

## 9. Demonstration readiness

### Safe offline demonstration

From `D:\Capstone Project`:

```powershell
python run_demo.py --task task_02 --seed 42 --mode offline --policy proposed --log "$env:TEMP\safe_intervention_demo.jsonl"
```

Use a fresh trace filename if that path already exists. The same command was executed with a unique temporary filename and passed. The CLI displays the selected decision, features, code, verifier evidence, and final result.

### Not verified

- Live Gemini provider connectivity is unverified. This audit made no provider request. Fake client tests prove local request construction/error handling only.
- Streamlit runtime is unverified because Streamlit is absent. The core CLI remains usable without it.
- The Phase 9.8 standalone pilot entrypoint was not run; doing so would rerun the frozen pilot and is prohibited here.
- This is an offline synthetic mechanics demonstration. Do not present it as a successful live LLM-agent experiment.

## 10. Confirmed bugs and ranked recommendations

1. **Low severity, input validation:** reject booleans in `run_demo.py:146`. It is a reproducible type-contract defect, though the current feature source supplies a float.
2. **Medium severity, test health:** replace the four stale v2/v3/v4 lock assertions noted above with temporary, version-correct lock fixtures. The full suite currently cannot be claimed green.
3. **Medium severity, reproducibility:** add a pinned dependency manifest consistent with supported Python versions; do not install dependencies as part of this audit.
4. **Documentation:** update the README `research_scope_status` table (around lines 115-130), which still labels pilot-test/prefix baselines planned despite later completed Phase 9.2/9.8 work. Keep Phase 9.2 and Phase 9.8 results distinct and preserve the document-status caveat already identified in the finalization checklist.
5. **Optional demonstration:** install Streamlit only through an approved package source if the UI itself is needed; otherwise use the verified CLI.
6. **Research integrity:** keep the missing full potential-outcomes archive, uncalibrated/synthetic risk semantics, privileged prefix comparator, no selected proposed interventions, and absence of held-out/live evidence visible in any presentation or paper.

## 11. Preservation and final verdict

This audit did not retrain models, run the Phase 9.8 pilot, regenerate experimental results, modify source code, alter locks/freezes, install packages, read `.env` contents, or make network/API calls. The current v4 lock bindings, freeze digest, and scored metrics were rechecked. The temporary CLI/test traces were isolated from project `logs/` and `results/`. The project has no `.git` directory, so a repository diff was unavailable; the audit's own project changes are limited to this new report.

**Final category: B — Ready after minor fixes.** The project is ready for a carefully labeled offline synthetic demonstration today. Before describing the whole repository as fully verified or submitting it as a live multi-agent result, update stale lock tests, correct the README status table, declare dependencies, and keep the live/dashboard limitations explicit. No code or historical artifact was changed during this health check.
