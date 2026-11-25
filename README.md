# Safe Intervention Without Rollback

This repository implements an offline deterministic task suite with Phase 3 baselines, Phase 4 proposed-policy training, Phase 5 soft repair, Phase 6 matched evaluation, an optional Phase 7 live Gemini path, and an optional Phase 8 Streamlit dashboard. Offline CLI and dashboard runs do not call Gemini. The live path uses the locked model and writes to a separate live JSONL log.

## Run

From the project root:

```powershell
python run_demo.py --task task_01 --seed 42 --mode offline
```

To demonstrate the Phase 3 threshold policy on a faulted scenario:

```powershell
python run_demo.py --task task_01 --seed 42 --mode offline --policy naive_threshold
```

Train the Phase 4 classifier and create the local model artifact and branch dataset:

```powershell
python -m policy.train_policy
```

Then make one proposed-policy decision from a task/seed prefix (no branch or rollback is run):

```powershell
python -m policy.predict_policy --task task_01 --seed 42
```

To demonstrate the Phase 5 proposed-policy trajectory, with repair applied in the current step:

```powershell
python run_demo.py --task task_01 --seed 42 --mode offline --policy proposed --log logs/phase5_demo.jsonl
```

The training command uses the first six task IDs for fitting and holds out `task_07` and `task_08` for evaluation. It prints the measured held-out score and class counts. The offline `failure_prob` input is currently a neutral 0.5 prior: this single-step suite has no independently validated pre-action signal, and scoring the corrupted instruction format would simply identify this fixture's fault injection. The classifier therefore has no risk signal to learn from. Its measured score is exploratory only and does not support generalization claims.

Run the Phase 6 evaluation with the saved model (this does not retrain it). By default, the runner reads held-out task IDs from model metadata (`task_07`, `task_08`):

```powershell
python -m experiments.run_comparison --seed-start 0 --seed-count 32 --output-dir results/heldout_evaluation
```

This command runs 2 held-out tasks with seeds 0–31 under all three policies (192 runs, 64 matched task/seed scenarios). `--tasks all` opts into all eight tasks, including the six training tasks; those results are exploratory and are not held-out evaluation. It generates:

- `results/heldout_evaluation/evaluation_runs.jsonl` — per-step and run-summary records, including scenario and initial-state hashes.
- `results/heldout_evaluation/comparison_summary.md` and `results/heldout_evaluation/comparison_metrics.csv` — metrics computed by rereading those run summaries.
- `results/heldout_evaluation/policy_comparison.svg` — grouped rate chart generated from the same computed metrics; it uses the Python standard library, so no plotting package is required.

Metric definitions: task success is verified passes / attempted trajectories; intervention count is the sum of live intervention events and intervention rate is trajectories with at least one / attempted trajectories; disruption is proposed/baseline interventions on matched cases where `no_intervention` passed / all such control successes (an intervention counts as disruption because it changed an already-successful trajectory); recovery is runs that intervene and pass / matched cases where `no_intervention` failed; average steps is executed steps / attempted trajectories, counting an aborted run as zero executed steps. Denominators are shown beside rates. `naive_threshold` still aborts iff risk is strictly greater than 0.5; with the current neutral prior of exactly 0.5, it continues.

Audited held-out result for the command above (64 scenarios):

| Policy | Success | Interventions | Intervention rate | Disruption | Recovery | Avg executed steps | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_intervention` | 32/64 (50.0%) | 0 | 0/64 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 1.000 | 0.000099 |
| `naive_threshold` | 32/64 (50.0%) | 0 | 0/64 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 1.000 | 0.000088 |
| `proposed` | 32/64 (50.0%) | 0 | 0/64 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 1.000 | 0.000811 |

After removing the fault-state feature leak, no policy intervened in this evaluation and the proposed policy matched the baselines: 32/64 verifier passes, zero verifier-backed recoveries, and one executed step per run. This is an honest result of the uninformative prior; it is not evidence of policy superiority. The model was fit on tasks `task_01`–`task_06` (192 examples, class counts 96/96) and evaluated on held-out tasks `task_07`–`task_08` (64 examples, class counts 32/32); measured held-out classifier accuracy was 0.500. Older artifacts at `results/evaluation_runs.jsonl`, `results/audit_rerun/`, `results/heldout_repeat/`, and `results/phase6_smoke/` predate the leakage correction and remain preserved for audit history; do not treat them as current corrected evaluation. The updated corrected trace and reports are in `results/heldout_evaluation/`.

Research validity note: the original offline risk feature leaked `scenario.fault_enabled` directly (0.8 for faulted, 0.2 for clean). It has been replaced in offline training and inference by an uninformative 0.5 prior because the available one-step fixtures do not justify a separate pre-action predictor. The classifier input contains only the required four features; the fault flag and oracle outcomes remain scenario/label/evaluation data, not policy features. The task suite, labels, and outcomes are still synthetic and tiny; the result does not validate a realistic critic. The live Gemini critic uses a different source and its feature distribution is not calibrated against this offline-trained classifier. Latency is local simulation runtime, not Gemini/API latency. No superiority or statistical-significance claim is supported.

The command returns exit code 1 when the injected faulty result fails the oracle, and 0 when the result passes. Choose any task ID from `task_01` through `task_08`; fault status is reproducible from task and seed. Trace events append to `logs/phase1.jsonl` by default; pass `--log path\to\trace.jsonl` to choose another file.

## Test

```powershell
python -m unittest discover -s tests -v
```

The environment used for Phase 1 has Python 3.13.16. Phase 1 uses only the Python standard library. `pytest` and `matplotlib` are not installed; no package installation was needed.

## Optional Streamlit dashboard

The optional dashboard has five sections: Project Overview, Interactive Execution, Decision Explanation, No-Rollback Demonstration, and Research Results. It calls the existing `run_demo.run` workflow and existing feature/policy components; it does not contain a separate policy engine. Interactive traces use unique files in the operating system temporary directory. Research Results reads the frozen Phase 9.8 metrics and audit report; it does not run an experiment or rewrite results.

The interactive page supports `no_intervention`, `naive_threshold`, and the **Phase 4 binary** `proposed` model. The displayed Phase 9.8 four-class cost rule is explicitly identified as a separate research model, not the Phase 4 CLI rule. Phase 9.8 values are loaded from the frozen pilot metrics and the primary estimate/interval are taken from the completed post-pilot audit report because the metrics JSON does not contain them.

The Phase 9.8 comparison is synthetic and exploratory. The proposed policy selected no interventions and tied `no_intervention`; the prefix comparator selected 232 counterfactual actions, none confirmed applied. `pilot_test` is descriptive and excluded from comparative metrics. This is not a live LLM experiment or evidence of real-world effectiveness. See the finalization reports for the missing potential-outcomes archive and other limits.

### Install and launch

From the project root in Windows PowerShell, create and activate an environment, then install the optional dashboard dependencies from `requirements-dashboard.txt` using your approved package source. The project core CLI does not require Streamlit.

```powershell
cd "D:\Capstone Project"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false
```

For a fully offline installation, provide the required wheels in a local wheelhouse and use `python -m pip install --no-index --find-links <wheelhouse> -r requirements-dashboard.txt`. No dependencies were installed while preparing the dashboard, so Streamlit runtime and UI rendering still require verification in an environment where the dependency is available.

The offline CLI demonstration remains available without Streamlit:

```powershell
python run_demo.py --task task_01 --seed 42 --mode offline --policy proposed --log "$env:TEMP\safe_intervention_demo.jsonl"
```

## Model restriction

The master specification locks all live roles to `gemini-3.8-flash` and prohibits automatic fallback, despite another stack mention of Gemini 3.5 Flash. Phase 7 implements the locked-model live path. It reads only the project's `.env`, requires an exact `GEMINI_MODEL=gemini-3.8-flash` if set, uses a 15-second request timeout and one SDK attempt, validates JSON responses, and logs sanitized errors without keys. `.env` is excluded by `.gitignore`; `.env.example` contains placeholders only. Do not infer credential validity or provider access from local configuration.

Before any future live request, check provider quota and billing: one single-task run makes one Planner planning request, one Critic and Coder request per planned step, and one Verifier request; an intervention adds a Planner replan request. Live errors use `logs/live_runs.jsonl`, separate from offline/evaluation logs; a live failure is never represented as a task pass. The deterministic task oracle remains authoritative even when the model verifier disagrees.

One authorized direct smoke command was attempted after Phase 7. Its sanitized terminal output was `model_id=gemini-3.8-flash`, `api_key_configured=True`, and `smoke=FAIL:configuration_or_request_error` with exit status 0. The wrapper hid the failure stage, so it is unknown whether the provider received the request. There is no successful provider-connectivity result. An offline-tested diagnostic helper now returns a nonzero status for future failures, but it has not been run against the provider.

Live demo command, after configuring the local key and approving any applicable usage:

```powershell
python run_demo.py --task task_01 --seed 42 --mode live --policy no_intervention
```

The live critic receives visible step/prefix information and is not passed `fault_enabled`. The offline feature is now a neutral 0.5 prior; see the updated held-out results and leakage correction above. Real-provider connectivity and the live single-task check remain unverified; no successful live experiment is claimed.

## research_scope_status

| Objective / feature | Status |
| --- | --- |
| Offline task pipeline and seeded fault | Implemented (eight tasks, Phase 2) |
| No-intervention and naive-threshold policies | Implemented (Phase 3; policy selection is per run) |
| Offline features and risk signal | Implemented (neutral prior; no validated task-suite signal available) |
| Proposed trained cost-sensitive policy | Implemented (Phase 4; current neutral features yield no intervention signal; exploratory only) |
| Offline labels and counterfactual branches | Implemented (Phase 4 training only) |
| Append-and-replan soft repair | Implemented (Phase 5; current-step intervention, no rollback) |
| Matched policy evaluation, metrics, and plots | Implemented (Phase 6; synthetic evaluation only) |
| Pilot-test and prefix-forecast baselines | Implemented in the frozen Phase 9.8 synthetic pilot; `pilot_test` is descriptive only |
| Feature ablation and cross-model transfer | Planned |
| Live Gemini mode | Implemented; provider verification pending project-local credentials and quota/billing approval |
| Metrics, chart, suite | Implemented (Phase 6; synthetic evaluation only) |
| Optional Streamlit dashboard | Implemented; optional dependency is absent in current environment, so UI runtime is not yet verified |

The current system is a mechanics demo, not a live LLM experiment or evidence of production policy effectiveness. The held-out task score must be interpreted with the documented synthetic-risk limitation. See `PROGRESS_LOG.md` for phase status and executed commands.

## Phase 8.3: separate synthetic fragility experiment

The original arithmetic benchmark remains unchanged and continues to use the
neutral `failure_prob=0.5` prior. The separate `experiments.fragility_experiment`
module tests whether a noisy, pre-action estimate of execution fragility can
support a cost-sensitive intervention decision under an explicit local
simulator. It uses no network, Gemini client, live mode, API key, or additional
dependency. Results are synthetic and do not establish realistic critic
effectiveness or real-world LLM-agent performance.

Run the experiment with new, separate output and model paths:

```powershell
python -m experiments.fragility_experiment --output-dir results/phase8_3_fragility_run --artifact models/phase8_3_fragility_run/proposed_policy.joblib
```

The runner refuses to overwrite a non-empty output directory or an existing
model artifact. Default generation uses root seed `20261028`, 128 replicates
per training task, and 128 per held-out task. It trains only on synthetic task
IDs 00–11 and evaluates on 12–15. To reproduce this recorded run, select fresh
output/model paths and keep the same seed and replicate settings.

The risk signal estimates execution fragility from `latent_fragility + seeded
Gaussian sensor noise`; its reported `failure_prob` also includes a fixed fault
prevalence prior. The actual injected-fault draw uses a separate random stream
and does not enter features. Feature order is the project schema:
`failure_prob`, `reversibility_proxy`, `trajectory_position`,
`disruption_history`. Only the first varies in this one-step synthetic
experiment. IDs, seeds, hashes, filenames, fault flags, and outcomes stay out of
the policy feature vector. See
[`results/phase8_3_fragility_final/scenario_design.md`](results/phase8_3_fragility_final/scenario_design.md)
and `experiments/fragility_experiment.py` for the generator and metric details.

Recorded held-out results from seed `20261028` (512 matched scenarios, 1,536
policy runs):

| Policy | Mean realized cost | Task success | Interventions | Aborts | Recovery | Average steps |
|---|---:|---:|---:|---:|---:|---:|
| no_intervention | 1.6230 | 235/512 (45.9%) | 0 | 0 | 0/277 | 1.000 |
| naive_threshold | 2.1914 | 138/512 (27.0%) | 0 | 272 | 0/277 | 0.469 |
| proposed | 1.6230 | 235/512 (45.9%) | 0 | 0 | 0/277 | 1.000 |

The proposed policy tied `no_intervention` on the primary cost and task success;
it made no interventions. It had lower measured cost than the aborting
`naive_threshold` on this simulator, but these results do not support a claim
of statistical significance or general superiority. The held-out risk Brier
score was `0.236060`; this is a descriptive score on this synthetic generator.
The shared execution draw and lower intervention failure threshold mean this
simulator cannot represent intervention-caused task harm, although it does
represent useful, unnecessary, and ineffective interventions. Task IDs are
IID strata from one generator rather than distinct generalization domains.
Reported runtime is local simulation time, not Gemini latency. Full measured
counts, reliability bins, raw JSONL logs, training examples, and machine-readable
metrics are under `results/phase8_3_fragility_final/`. The earlier same-seed
first run is preserved separately in `results/phase8_3_fragility/`; the `_final`
folder is the run whose raw logs and metrics passed the read-only audit.

Audit the recorded run without changing files:

```powershell
python -m experiments.fragility_experiment --audit --output-dir results/phase8_3_fragility_final --artifact models/phase8_3_fragility_final/proposed_policy.joblib
```

The original CLI, optional dashboard, saved Phase 4 model, and earlier
evaluation artifacts are preserved. This experiment has its own audited
artifact at `models/phase8_3_fragility_final/proposed_policy.joblib` (and the
first-run artifact at `models/phase8_3_fragility/proposed_policy.joblib`) and does not modify the
original benchmark's risk feature or proposed policy.

## Phase 8.6: proposed-policy decision diagnostics

Future Phase 8.3 offline proposed-policy run records add an optional
`decision_diagnostics` object with `p_helpful`, `continue_expected_cost`,
`intervene_expected_cost`, `cost_difference`, and `selected_action`. These are
the classifier's predicted helpful-intervention probability and expected costs
under the existing matrix; they are not realized evaluation costs. The latter
remain computed from completed outcomes by the existing metrics path.

The strict action rule is unchanged: intervene only when `p_helpful > 0.5`; at
exactly `0.5`, costs tie at `1.5` and the policy continues. `p_helpful` is the
classifier's estimate that intervention will help. It is distinct from the
`failure_prob` feature, which is the synthetic noisy pre-action failure-risk
estimate supplied to the classifier. Invalid or malformed classifier
probabilities raise `InvalidHelpfulProbability` and fail the run instead of
being recorded as a valid continuation.

The diagnostics field is additive. Existing JSONL records without it remain
readable, and baseline records use `null`. This change adds observability only;
it does not validate the saved model, change performance, or regenerate prior
experiment results.

## Phase 9: four-class cost-sensitive synthetic experiment

Phase 9 is isolated in `experiments/phase9_cost_sensitive.py`; it does not
change the Phase 8.3 experiment, Phase 4 classifier, CLI, dashboard, or live
Gemini execution. The preregistration is written to
`results/phase9_cost_sensitive_preregistration.md` before held-out potential
outcomes are generated. The primary output and model paths are new:

```powershell
python -m experiments.phase9_cost_sensitive --mode primary --output-dir results/phase9_cost_sensitive --artifact models/phase9_cost_sensitive/proposed_policy.joblib --preregistration results/phase9_cost_sensitive_preregistration.md
```

The primary design trains on 12 disjoint synthetic family IDs × 128 examples
and evaluates on 20 held-out family IDs × 128 replicates × five fixed root
seeds (12,800 matched scenarios). Family IDs are group keys from one generator,
not distinct real-world domains. The four classes are recovery, unnecessary
intervention, ineffective intervention, and intervention harm. Frozen costs
are continue `3/0/3/0` and intervene `2/2/5/5` in that class order. Failure
penalty `F=3` and intervention overhead `O=2` are included once; aborts remain
failures with cost 3. The 10% harm probability is an illustrative synthetic
assumption, not an empirical rate.

Policy decision records and counterfactual evaluation-only potential outcomes
are stored separately in `policy_runs.jsonl` and
`evaluation_potential_outcomes.jsonl`. All policy decisions are fixed before
the held-out potential outcomes are recorded. Metrics include four-class
probability scores, reliability, class confusion metrics, matched policy
outcomes, and family-cluster bootstrap intervals. Potential outcomes are
simulator-generated counterfactuals, not simultaneously observed real-world
branches. These results are exploratory and do not establish real-world
LLM-agent effectiveness or statistical power.

Audit the recorded outputs without deserializing the model:

```powershell
python -m experiments.phase9_cost_sensitive --mode audit --output-dir results/phase9_cost_sensitive --artifact models/phase9_cost_sensitive/proposed_policy.joblib
```

Optional preregistered harm-rate stress tests reuse the primary-trained model
and have separate output directories:

```powershell
python -m experiments.phase9_cost_sensitive --mode sensitivity --harm-probability 0.0 --output-dir results/phase9_cost_sensitive_harm_000 --artifact models/phase9_cost_sensitive/proposed_policy.joblib --preregistration results/phase9_cost_sensitive_preregistration.md
python -m experiments.phase9_cost_sensitive --mode sensitivity --harm-probability 0.05 --output-dir results/phase9_cost_sensitive_harm_005 --artifact models/phase9_cost_sensitive/proposed_policy.joblib --preregistration results/phase9_cost_sensitive_preregistration.md
python -m experiments.phase9_cost_sensitive --mode sensitivity --harm-probability 0.2 --output-dir results/phase9_cost_sensitive_harm_020 --artifact models/phase9_cost_sensitive/proposed_policy.joblib --preregistration results/phase9_cost_sensitive_preregistration.md
```

Sensitivity runs are stress tests; their changed outcome distribution is not
used to select a model or policy. A classifier trained at 10% harm is not
assumed calibrated at other harm rates. Never point these commands at existing
Phase 8.7 or historical output/model paths.

### Recorded Phase 9 results

The primary run used 1,536 training examples (class counts: recovery 397,
unnecessary 639, ineffective 421, harm 79) and 12,800 held-out scenarios.
Held-out potential-outcome counts were recovery 3,184, unnecessary 5,461,
ineffective 3,551, and harm 604. The independent 10% harm-event draw occurred
in 1,286 scenarios; only 604 produced the defined intervention-harm outcome
because that category also requires continuation to pass and intervention to
fail.

| Policy | Success | Mean realized cost | Interventions | Recovery | Harm selected | Avg. steps |
|---|---:|---:|---:|---:|---:|---:|
| no_intervention | 6,065/12,800 (47.4%) | 1.5785 | 0 | 0 | 0 | 1.000 |
| naive_threshold | 3,679/12,800 (28.7%) | 2.1377 | 0 | 0 | 0 | 0.462 |
| proposed | 6,065/12,800 (47.4%) | 1.5785 | 0 | 0 | 0 | 1.000 |

The proposed classifier continued in all held-out cases. Its mean predicted
probabilities (recovery, unnecessary, ineffective, harm) were
0.2576/0.4180/0.2727/0.0517, versus observed class frequencies
0.2488/0.4266/0.2774/0.0472. Multiclass Brier was 0.64594 and log loss
1.16062. Mean expected costs were 1.59085 to continue and 2.97301 to
intervene. Continue was cheaper in all 12,800 cases; the intervene-minus-
continue expected-cost difference ranged from 1.0793 to 1.7237. The proposed policy therefore tied no_intervention because it
selected the same action in every scenario; it did not demonstrate intervention
effectiveness. Proposed-minus-no_intervention mean cost difference was 0.0000
(95% family-cluster bootstrap interval [0.0000, 0.0000]); proposed-minus-
naive_threshold was -0.5592 (interval [-0.5822, -0.5370]). These intervals are
exploratory and do not establish statistical significance or generalization.

| Synthetic harm assumption | Potential harm outcomes | Proposed mean cost | Proposed interventions | Multiclass Brier | Log loss |
|---:|---:|---:|---:|---:|---:|
| 0% | 0/12,800 | 1.5785 | 0 | 0.60893 | 1.06350 |
| 5% | 299/12,800 (2.34%) | 1.5785 | 0 | 0.62738 | 1.11153 |
| 10% primary | 604/12,800 (4.72%) | 1.5785 | 0 | 0.64594 | 1.16062 |
| 20% | 1,188/12,800 (9.28%) | 1.5785 | 0 | 0.68132 | 1.25427 |

All policies continued or aborted according to their fixed rule; proposed never
intervened in these runs. Sensitivity runs reused the primary model and were
not used for model or policy selection. The observed 10% event rate is a
simulator parameter, not a measured real-world intervention-harm rate.

## Phase 9.2: diverse synthetic mechanisms and stronger evaluation

Phase 9.2 is isolated in `experiments/phase9_2_simulator.py` and
`experiments/phase9_2_evaluation.py`; earlier phases and artifacts remain
unchanged. The experiment is offline and synthetic, not a live LLM-agent
evaluation.

The frozen split uses 12 family configurations × 256 training scenarios, 6 ×
256 validation, 4 × 256 separate pilot, and 30 × 256 held-out scenarios
(7,680 matched test scenarios). It includes eight distinct synthetic outcome
mechanisms. The SHA-256 split manifest is frozen before held-out potential
outcomes are generated. Inference uses only the ordered four-feature schema:
`failure_prob`, `reversibility_proxy`, `trajectory_position`, and
`disruption_history`. Hidden state, realized fault, IDs, seeds, labels, and
potential outcomes are excluded from the feature dictionary. Random streams
are deterministic and domain-separated.

Exact offline pipeline commands:

```powershell
python -m experiments.phase9_2_evaluation --mode prepare
python -m experiments.phase9_2_evaluation --mode train
python -m experiments.phase9_2_evaluation --mode verify-artifacts
python -m experiments.phase9_2_evaluation --mode evaluate
python -m experiments.phase9_2_evaluation --mode audit
```

`train` uses train/validation/pilot data only and saves the full model and five
ablation models in `models/phase9_2_diverse_final/`. `verify-artifacts` checks
the saved model checksums, fitted pipeline, feature order, and class order.
`evaluate` loads the saved classifier without retraining and writes matched
records to `results/phase9_2_diverse_final/`. `audit` recomputes metrics from
the JSONL files and checks the saved model hash without deserializing it. All
commands refuse to overwrite their frozen design or existing output paths.

The primary outcome is paired mean realized cost, with a 95% family-configuration
bootstrap interval (30 clusters, 10,000 resamples). Class-probability quality,
per-class scores, mechanism groups, feature ablations, and matched policy
results are reported as secondary descriptions. The prefix-branching comparator
uses eight extra simulator forecasts per scenario and has more information and
compute than the learned policy; it is an offline diagnostic, not a deployable
matched-budget baseline. The pilot baseline selects its action on the separate
pilot set.

Sensitivity runs reuse the frozen 10%-harm model and use separate new folders:

```powershell
python -m experiments.phase9_2_evaluation --mode sensitivity --harm-probability 0.0 --output-dir results/phase9_2_harm_000
python -m experiments.phase9_2_evaluation --mode audit --output-dir results/phase9_2_harm_000 --harm-probability 0.0
python -m experiments.phase9_2_evaluation --mode sensitivity --harm-probability 0.05 --output-dir results/phase9_2_harm_005
python -m experiments.phase9_2_evaluation --mode audit --output-dir results/phase9_2_harm_005 --harm-probability 0.05
python -m experiments.phase9_2_evaluation --mode sensitivity --harm-probability 0.2 --output-dir results/phase9_2_harm_020
python -m experiments.phase9_2_evaluation --mode audit --output-dir results/phase9_2_harm_020 --harm-probability 0.2
```

### Recorded Phase 9.2 results

The primary evaluation recorded 7,680 matched scenarios and 38,400 policy runs.

| Policy | Verifier successes | Mean realized cost | Interventions | Recovery / unnecessary / ineffective / harm |
|---|---:|---:|---:|---:|
| no_intervention | 3,812/7,680 (49.64%) | 1.51094 | 0 | 0 / 0 / 0 / 0 |
| naive_threshold | 2,849/7,680 (37.10%) | 1.88711 | 0 (2,698 aborts) | 0 / 0 / 0 / 0 |
| proposed | 3,812/7,680 (49.64%) | 1.51094 | 0 | 0 / 0 / 0 / 0 |
| pilot_test | 3,812/7,680 (49.64%) | 1.51094 | 0 | 0 / 0 / 0 / 0 |
| prefix_branching_approximation | 3,994/7,680 (52.01%) | 1.56510 | 481 | 197 / 55 / 214 / 15 |

The learned policy continued in all test scenarios and exactly tied
`no_intervention`. It did not demonstrate selected intervention effectiveness.
Its multiclass Brier score was 0.681570 and log loss 1.255743. Proposed minus
no-intervention mean cost was 0.000000 (95% family bootstrap interval
[0.000000, 0.000000]); proposed minus naive was -0.376172
([-0.398047, -0.352734]). These exploratory intervals are not claims of
statistical significance. Naive threshold aborts remain failures; its 2,698
aborts are not repair interventions.

At 0%, 5%, and 20% illustrative intervention-harm settings, proposed again
made zero interventions and retained cost 1.51094 and success 3,812/7,680.
The prefix-branching diagnostic had mean costs 1.56589, 1.56510, and 1.55326,
with 709, 580, and 293 interventions respectively. These are sensitivity
results from the synthetic simulator, not empirical harm rates. At the primary
10% setting, the comparator recorded 15 intervention-harm outcomes.

Primary generated files include `evaluation_runs.jsonl`,
`evaluation_potential_outcomes.jsonl`, `evaluation_metrics.json`,
`ablation_runs.jsonl`, `ablation_metrics.json`, `comparison_summary.md`,
`scenario_design.md`, and `evaluation_run_manifest.json` under
`results/phase9_2_diverse_final/`. Sensitivity logs and metrics are separate
under their corresponding harm-rate folders.

Limitations: all mechanisms and risk signals are synthetic assumptions;
`failure_prob` is fixture-derived and not a validated critic; the learned
policy's all-continue behavior provides no evidence of intervention
usefulness; and there are only 30 held-out family configurations. The
prefix-branching comparator has simulator access and extra compute. Results do
not support real-world generalization, statistical-significance, or live
LLM-agent effectiveness claims. The recommended next step is an independent
research-design review for empirically justified observable risk signals
before defining a new held-out experiment.

## Phase 9.7: isolated observation and runner integration

Phase 9.7 adds `experiments/phase9_7_simulator.py` and
`experiments/phase9_7_runner.py` as a separately versioned offline path. The
runner creates policy decisions from a public, provenance-checked snapshot and
passes only the ordered four-feature mapping to an injected Phase 9.7 policy
adapter. Evaluator truth is stored separately and is never passed through the
policy interface. The runner does not load a model, train a model, write an
experiment result, or run a policy comparison. Phase 9.2 and older paths are
unchanged.

The frozen `synthetic-prefix-monitor-v1` score is
`0.25*anomaly_score + 0.25*alert + 0.25*min(errors,2)/2 +
0.25*min(warnings,2)/2`. It is an **uncalibrated synthetic risk score**, not a
calibrated probability or observed critic result. A shared synthetic
susceptibility drives noisy pre-decision sensor measurements and separate
simulated outcomes. Sensor noise, alerts, visible counts, plan/action metadata,
history, continuation execution, repair, and intervention harm use distinct
seeded streams. This sensor-to-outcome relationship and all numeric
mechanism parameters are simulator assumptions, not empirically validated
evidence, and were not tuned against held-out outcomes.

Snapshots require `observed_through_sequence = decision_sequence - 1`, matching
logical evidence/decision ticks, a complete contiguous event prefix, and a
matching prefix digest. Only confirmed prior `intervention_applied` events
contribute to `disruption_history`. Later ledger entries do not mutate an
earlier immutable snapshot. The three policy names and the strict
`failure_prob > 0.5` naive-abort rule are retained; the threshold applies to
the uncalibrated score on this new path.

Focused integration tests:

```powershell
python -m unittest tests.test_phase9_7_integration -v
```

The Phase 9.7 tests exercise fixed mechanism fixtures and integration
contracts only. They are not a training run, held-out evaluation, policy
comparison, or evidence of policy effectiveness. A separately versioned model
training phase and a new preregistration/split manifest are required before
any held-out outcomes are generated. Existing results must not be combined
with the changed feature semantics.
