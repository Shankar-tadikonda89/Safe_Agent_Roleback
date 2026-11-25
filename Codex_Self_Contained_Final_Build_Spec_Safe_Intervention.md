# Codex Master Build Specification

## Safe Intervention Without Rollback

**Purpose:** Build a reliable, demonstrable academic capstone as quickly
as possible without sacrificing correctness, reproducibility, or
research integrity.

**This is the complete, self-contained build specification.** It consolidates the requirements, methodology, architecture, constraints, tools, evaluation plan, and research extensions from the project brief, Review 1 report, and Review 1 presentation. Codex must be able to implement the project using this document alone. The user is not required to provide or upload any other project documents. Do not stop or ask for the original documents merely because they are absent.

Treat this specification as the authoritative implementation contract. If the user later provides a newer requirement, identify any conflict and ask before changing a locked requirement. Preserve the research intent and report implementation limitations honestly; never silently rewrite academic claims to make results look stronger.

------------------------------------------------------------------------

## Document authority and completeness

This file is the sole required handoff artifact for Codex. All architecture, implementation requirements, tools, model restrictions, test criteria, experiment scope, reporting rules, and phase gates needed for the MVP are specified below. Do not ask the user to upload the original brief, report, or presentation to begin. If a detail is genuinely unspecified, choose the simplest safe implementation consistent with this document, record the assumption in `PROGRESS_LOG.md`, and ask only when the decision affects a locked requirement, research validity, destructive action, or cost.

------------------------------------------------------------------------

## 1. Mission and success criteria

Build a working, reproducible system for the capstone:

**Safe Intervention Without Rollback: A Lightweight, Rollout-Free
Decision Policy for Proactive Failure Prevention in LLM Multi-Agent
Systems.**

Research question:

> Can a lightweight, single-pass policy decide when to intervene in a
> multi-agent trajectory, using cheaply observable features instead of
> counterfactual re-execution at decision time, while reducing
> unnecessary disruption?

The central contribution is **a per-trajectory, cost-sensitive
intervention decision policy** placed between a failure-risk signal and
a multi-agent executor. It is not merely a multi-agent chatbot, a
failure detector, or a dashboard.

The top priority is to get a real end-to-end demo working early, then
expand its research scope. Do not attempt to build every research
ambition in one pass. The system must be honest about which results are
synthetic, which use live LLM calls, which baselines are actually
reproduced, and which objectives remain future work.

### Definition of a minimum successful demo

A user must be able to run a documented command and see: 1. A selected
task and deterministic fault/seed. 2. The Planner → Coder → Verifier
trajectory. 3. A failure-risk estimate and the features used for the
decision. 4. Results for `no_intervention`, `naive_threshold`, and
`proposed`. 5. Identical task configuration, seed, initial state, and
fault condition across strategy comparisons. 6. An explicit intervention
trace, including corrective instruction and replanning from the current
step. 7. Final outcomes and computed metrics, derived from actual run
logs. 8. A chart of failure risk with the intervention point marked. 9.
Automated tests demonstrating reproducibility and no live rollback. 10.
An offline deterministic mode that works without an API key or network.

The demo should be operable in under 10 minutes. A clean CLI is the
first priority. Add a Streamlit interface only after the CLI, core
logic, tests, and metrics work.

**Never promise "without any errors."** Instead, reduce risk through
small milestones, tests, clear failure messages, checkpoints, and a
rehearsed offline demo.

------------------------------------------------------------------------

## 2. Resolve source-document conflicts without stalling progress

The source documents describe two levels of scope. Use them deliberately
rather than mixing them accidentally.

### Track A --- Reliable demo MVP (mandatory first)

Use a compact synthetic task suite of 8--12 coding/math tasks,
deterministic seeded fault injection, a shared three-agent pipeline,
three policies, offline counterfactual label generation for training
only, a small scikit-learn classifier, JSONL logs, metrics, and a CLI.
Use a plain Python loop unless a framework clearly reduces complexity.
No database, cloud deployment, or heavyweight benchmark integration is
required for this MVP.

The detailed constraints in `ANTIGRAVITY_PROJECT_BRIEF.md` apply to this
track.

### Track B --- Research extensions (after Track A passes acceptance tests)

The Review 1 report additionally commits to: - Four baselines: no
intervention, naive threshold, pre-deployment pilot-test policy, and a
reproduced prefix-branching controller. - Evaluation of the cost/quality
trade-off against prefix branching. - Feature ablation. - Cross-model
generalization across at least two base LLMs. - Potential use of
existing failure benchmarks and more extensive experiments.

Implement these only after the MVP is stable and only when time, access,
and compute permit. Maintain a `research_scope_status` section in the
README and progress log distinguishing **implemented**, **partially
implemented**, **not implemented**, and **planned**. Do not claim the
MVP has fulfilled Track B objectives if it has not.

### Model conflict and API budget

The original build requirements lock the live model to **`gemini-3.8-flash` exclusively, with no automatic fallback**, while another technology-stack mention names Gemini 3.5 Flash. Follow the explicit locked-model rule: use `gemini-3.8-flash` for every live role (Planner, Coder, Verifier, critic/risk estimator, and any LLM-based feature source), do not silently substitute another model, and document the inconsistency in the README. If the provider rejects the model or the account cannot access it, show a clear error and preserve the deterministic offline demo. Do not change the model, `GEMINI_MODEL`, or `.env` without telling the user and obtaining explicit confirmation. Never create an automatic fallback chain or use a third-party model.

The source brief warns that free-tier daily request quotas may be tight (it cites 20 requests/day for the specified model at the time it was written). Treat that figure as a source-document estimate, not a guaranteed current quota; provider quotas can change. Before any live API request, inspect only this project's `.env` for a configured key. If the key is missing, continue offline and ask the user to configure it; never search outside the project folder for credentials. If a key is present, explain that the probe consumes quota and ask the user whether to run one small connectivity probe and whether billing is enabled. Do not assume either answer. Never run the full suite against the live API as the first test.

------------------------------------------------------------------------

## 3. Research scope and core concepts

### Multi-agent testbed

Use three roles: - **Planner:** the only agent that sees the complete
original task description. It decomposes the task into 2--4 concrete
steps and can re-plan when given corrective guidance. - **Coder:**
receives only the current step instruction plus code accumulated from
earlier steps in that run. It must not receive the full original task
prompt on every step. This constraint is important so a corrupted step
instruction cannot be silently corrected by reconstructing the canonical
textbook solution from the task description. - **Verifier:**
executes/checks the produced result against a deterministic
task-specific oracle where possible and returns structured pass/fail
evidence.

A plain function-call loop is preferred for the MVP. LangGraph is
optional, not mandatory. Avoid unnecessary agent frameworks and
abstractions.

### Failure critic

The critic returns a numeric `failure_prob` in `[0, 1]` for the current
trajectory prefix/step. Validate the output and handle malformed or
missing values explicitly. In live mode, the critic may be an LLM-based
risk assessor using a structured prompt; in offline mode, use a
deterministic, transparent risk estimator tied to known task/fault state
or a clearly labelled fixture.

Do not claim to reproduce AgentForesight or fine-tune its model unless
the implementation genuinely does so. Describe the MVP critic accurately
as a demo-scale risk estimator inspired by prefix-monitoring work.
Record the critic source/mode in every run.

### Required features

At minimum, build the four features in the build brief: 1.
`failure_prob` 2. `reversibility_proxy` 3. `trajectory_position` 4.
`disruption_history`

The report also discusses critic confidence/calibration, semantic drift,
and action type. Design the feature extractor so these can be added in
the research extension without breaking the MVP. Document which features
are active in each experiment. Avoid leakage: features used for a
decision must be observable from the trajectory prefix at that moment
and must not include future outcomes or offline labels.

`reversibility_proxy` must be based on documented action types/metadata
(e.g. read-only vs state-changing); do not pretend a simple heuristic is
a validated real-world reversibility measure. `disruption_history` must
be computed only from information available by the current step. If
there has been no prior intervention, use a documented default such as
0.0.

### Policies to implement

1.  **`no_intervention`:** ignore the risk score and continue.
2.  **`naive_threshold`:** if `failure_prob > 0.5`, hard-stop/abort with
    no repair, matching the build brief. This is intentionally a simple
    baseline; report that its task completion can collapse because it
    aborts.
3.  **`proposed`:** a saved, trained, cost-sensitive scikit-learn
    classifier predicts `continue` or `intervene` from the feature
    vector. On intervention, append corrective guidance to the Planner
    and re-plan from the current step. Do not rewind or restore the live
    trajectory to a previous state.

Keep strategy names and semantics stable across code, CLI, logs,
dashboard, and README.

### No rollback invariant

In live proposed mode: - Never restore a previous state snapshot to
continue execution. - Never re-run a candidate continuation from a saved
prefix to choose the current action. - Never use a future outcome,
offline branch result, or label in the live feature vector. -
Intervention is append/correct/replan from the current state. - Offline
branching is permitted only for label generation and the explicit
prefix-branching baseline, not as part of live proposed inference.

Make this invariant testable. Add an event/trace field indicating
whether an event is an offline counterfactual branch or a live action.
Add a test that fails if the proposed live path invokes the
branching/rollback helper.

------------------------------------------------------------------------

## 4. Fault injection and task suite

Create 8--12 short coding/math tasks with deterministic expected outputs
and task-specific validators. About half should have seeded faults at
known steps. Use a `--seed` option and make the fault reproducible. A
fault may be a deliberately misleading Planner step instruction or a
controlled corrupted step prompt. Do not rely on random LLM mistakes to
create every failure.

Each task definition should include: - Stable task ID and human-readable
title. - Full task prompt (Planner-only). - Input/initial state and
expected output or validator. - Number/description of steps. - Fault
injection enabled/disabled, fault step, and fault type. - Seed
handling. - Safe execution constraints.

Use isolated temporary directories for generated code/files. Never
execute arbitrary untrusted generated shell commands. If generated
Python code is run, execute it in a controlled subprocess with timeout,
restricted working directory, captured output, and clear errors. Prefer
task-specific function invocation and deterministic validators over
unrestricted execution.

A seeded fault should be meaningful: it should make the baseline's
result incorrect or incomplete in a reproducible way, and the proposed
intervention should have a plausible way to correct it. Do not rig the
evaluator to force the proposed policy to win. Include tasks where
intervention is unnecessary or harmful, so disruption is measurable.

------------------------------------------------------------------------

## 5. Offline training labels and classifier

Training data construction may branch counterfactually **offline
only**: 1. Capture a trajectory prefix and its available features. 2.
From the same prefix and initial conditions, evaluate a `continue`
continuation and an `intervene` continuation in isolated offline
branches. 3. Define the positive label clearly. The MVP label may be
`intervene_helped = 1` only when intervention passes and continuation
fails; document that this simple binary label does not capture every
cost trade-off. 4. Record both branch outcomes, branch costs, seed, task
ID, and prefix ID. 5. Train a small classifier, initially
`LogisticRegression` with `class_weight` or sample weighting as
appropriate. A tiny MLP is optional only if it demonstrably improves
results. 6. Save the complete fitted preprocessing/model pipeline to a
versioned artifact using `joblib`. 7. Load that artifact for live
proposed inference and call `predict_proba`/`predict` once per decision.
8. If training data has only one class, too few samples, or invalid
values, stop with a clear explanation. Do not fabricate a model or
silently replace it with a hard-coded `if/else` policy.

Use an explicit cost matrix or documented sample/class weights to
penalize unnecessary disruption more than a missed intervention where
appropriate. State how costs were selected. Keep the policy's action
threshold and training configuration in a config file. The proposed
decision must be classifier-driven in the primary experiment, not a
disguised manually tuned rule.

### Data leakage prevention

Split by task/trajectory/prefix group, not by randomly splitting
near-duplicate prefixes into train and test. Do not fit preprocessing on
test data. Keep a held-out evaluation set. If the tiny synthetic suite
cannot support a statistically meaningful split, state that limitation,
use repeated seeded trials, and treat results as exploratory.

------------------------------------------------------------------------

## 6. Required baselines and research extensions

### MVP baseline comparison

Run these three policies on the same task/seed/config: -
`no_intervention` - `naive_threshold` - `proposed`

Each policy should start from a fresh, equivalent initial state, with
the same task, seed, and injected fault. The system should preserve a
shared scenario manifest/hash to demonstrate matched conditions. For
live LLM runs, model outputs can vary; record model ID, prompt/template
version, temperature, seed where supported, and timestamp. Do not claim
exact determinism from a nondeterministic provider.

### Research extensions, after MVP is stable

Add the following in order: 1. **Pre-deployment pilot-test policy:**
estimate whether intervention should be enabled for a model/task setting
from a separate pilot set, then freeze that decision for evaluation. 2.
**Prefix-branching controller baseline:** a clearly separated offline
controller that compares continue/intervene from identical prefixes. Do
not integrate this into live proposed mode. 3. **Feature ablation:**
compare risk-only features against the full available feature set. 4.
**Cross-model transfer:** train on one supported model's trajectories
and evaluate on a second without retraining. First check whether the
second model is accessible and affordable. Do not violate the locked
model for the MVP; obtain explicit user approval before adding another
model. 5. **Existing benchmark integration:** only if the dataset is
legally/technically available and can be used within the schedule. Keep
synthetic MVP tasks as the reproducible smoke test.

Do not label the prefix-branching baseline "faithfully reproduced"
unless its algorithm, data protocol, and evaluation match the paper
sufficiently and are documented. Otherwise call it an approximation.

------------------------------------------------------------------------

## 7. Metrics and honest reporting

Compute all metrics from actual logged runs; never hard-code sample
numbers or force a target result.

Required MVP metrics: - **Task success rate:** completed tasks that pass
the task oracle / attempted tasks. - **Disruption rate:** among cases
that would have succeeded without intervention, the fraction in which
intervention was unnecessary and changed/disrupted the outcome. State
the exact operational definition used. - **Recovery rate:** among cases
that were failing before intervention, fraction recovered to a passing
outcome after intervention. - **Average steps:** mean executed steps,
with aborted runs treated consistently and explained. - **Intervention
count/rate:** number/fraction of trajectories or steps with
intervention. - **Inference cost:** LLM calls, approximate tokens when
available, wall-clock time and errors. Distinguish policy inference cost
from total agent calls.

For the research extension: - Cost-adjusted comparison against prefix
branching. - Feature ablation. - Cross-model generalization. -
Confidence intervals or variability across repeated seeds where
feasible.

Always show denominator/sample count beside percentages. Save raw
per-run records and an aggregate summary. If proposed performs worse
than a baseline, report it honestly and diagnose why. Never manipulate
tasks, metrics, or logs to obtain a desired ranking. The illustrative
metrics in the brief are examples only, not acceptance targets.

------------------------------------------------------------------------

## 8. Logging and artifacts

Write JSONL logs with one structured event per step and a run summary.
Include, as applicable: - `run_id`, timestamp, mode/policy, task ID,
seed, scenario hash - model ID and agent role - step index and
instruction ID - fault enabled/type/location - current code/result/check
evidence - `failure_prob` and critic source - feature values used by the
policy - decision (`continue`, `intervene`, or `abort`) - intervention
reason and corrective instruction - branch type (`live` or
`offline_counterfactual`) - verifier outcome and final task result -
latency, call counts, errors and token usage if supplied by provider

Do not log secrets or the API key. Keep generated logs, model artifacts
and local `.env` out of Git by default; include a small, non-sensitive
example config and `.env.example`. If representative logs are committed,
label them as fixtures/sample data.

Recommended structure:

``` text
safe-intervention-without-rollback/
├── agents/
│   ├── planner.py
│   ├── coder.py
│   ├── verifier.py
│   └── critic.py
├── environment/
│   ├── task_runner.py
│   ├── fault_injector.py
│   └── evaluator.py
├── policy/
│   ├── features.py
│   ├── generate_labels.py
│   ├── train_policy.py
│   └── decision_policy.py
├── modes/
│   ├── baseline.py
│   └── proposed.py
├── tasks/
│   └── task_suite.json
├── experiments/
│   ├── run_comparison.py
│   └── metrics.py
├── dashboard/
│   └── app.py                 # add only after CLI works
├── tests/
├── logs/                      # ignored, created at runtime
├── models/                    # local trained artifacts; document generation
├── results/                   # ignored or generated
├── run_demo.py
├── visualize_trajectory.py
├── requirements.txt
├── .env.example
├── .gitignore
├── PROGRESS_LOG.md
└── README.md
```

Adapt the structure if there is a good reason, but keep the core
separation between agents, task execution, policy training, live
inference, and evaluation.

------------------------------------------------------------------------

## 9. Technology and credentials

-   Python 3.11+.
-   `google-genai` for Gemini live mode.
-   `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `joblib`,
    `python-dotenv`, `pytest`.
-   `streamlit` optional, after CLI works.
-   No database required.
-   No cloud server required.
-   No GPU required for the MVP.

Use a `.env` file with `GEMINI_API_KEY` and a documented
`GEMINI_MODEL=gemini-3.8-flash` default, but do not overwrite an
existing `.env`. Keep `.env` in `.gitignore`. Track `.env.example`
containing placeholders only. Do not search home directories, editor
settings, sibling folders, shell history, or the broader filesystem for
keys. If the key is absent, ask the user to configure it.

Live API failures (401/403, 404, 429, 5xx, timeout, malformed output)
must result in a useful error message and non-zero exit code for the
live command. Do not silently switch models. The offline deterministic
mode must still work without the API.

Use short timeouts and limited retries only for transient failures, with
a small retry cap and visible logging. Do not automatically retry
quota/permission/model-not-found errors. Never run a full suite against
the live API as the first test. Start with one request and one task,
after user confirmation.

------------------------------------------------------------------------

## 10. Development protocol: work fast, but validate every milestone

Do not generate the whole repository in one enormous unverified edit.
Work in small, complete slices and run tests after each slice.

### Phase 0 --- Inspect and plan

-   Inspect the current working folder and repository files only; this specification is the complete requirements source.
-   Identify existing code and preserve useful work before editing.
-   Explain conflicts or implementation risks found within the specification and the intended resolution. Do not invent missing requirements.
-   Create/refresh `PROGRESS_LOG.md`.
-   Provide a short implementation plan and acceptance-test checklist.
-   Then begin Phase 1 without waiting for another approval unless a consequential decision is required (model change, paid service, destructive edit, or missing credential).

### Phase 1 --- Offline vertical slice

-   Create the smallest complete task pipeline.
-   Implement one deterministic task, one seeded fault,
    Planner/Coder/Verifier interfaces, task oracle, and JSONL trace.
-   Ensure the Coder sees only the current step instruction and
    accumulated code, never the full original prompt.
-   Add tests and run them.
-   Acceptance: one command runs end-to-end without network/API access
    and produces a truthful pass/fail result and trace.

### Phase 2 --- Task suite and reproducibility

-   Add 8--12 tasks and seeded fault injection.
-   Verify same task/seed reproduces the same fault and scenario hash.
-   Test both faulted and non-faulted cases.
-   Acceptance: narrow tests pass before full suite.

### Phase 3 --- Critic and baseline policies

-   Add risk estimate and logging.
-   Implement `no_intervention` and `naive_threshold` exactly as
    defined.
-   Run matched comparisons.
-   Acceptance: per-step decisions and metrics come from real logs.

### Phase 4 --- Offline labels and learned policy

-   Implement isolated counterfactual branches for training only.
-   Generate labels, inspect class balance, split by task/trajectory
    group, train and save a classifier.
-   Implement one-pass live inference from the saved model.
-   Add tests proving the proposed live path cannot call branch/rollback
    functions.
-   Acceptance: model can be deleted and regenerated by documented
    commands; live proposed inference loads it and does not branch.

### Phase 5 --- Soft repair

-   On `intervene`, append corrective context to the Planner and re-plan
    from the current step.
-   Preserve the existing live execution history; do not reset or rewind
    it.
-   Verify that a repair can help at least one valid test case and that
    no-intervention remains a fair control.
-   Include cases where intervention is not beneficial.
-   Acceptance: trace proves where the intervention happened and what
    changed.

### Phase 6 --- Experiment runner and visualization

-   Run all three strategies against matched scenarios.
-   Compute success, disruption, recovery, average steps, intervention
    rate and cost.
-   Generate a trajectory chart with risk and intervention point.
-   Acceptance: aggregate metrics can be regenerated from JSONL logs.

### Phase 7 --- Live Gemini integration

-   Only after the offline system works, add the locked model using the
    documented SDK.
-   Make one minimal connectivity test and one single-task test before
    wider use.
-   Record actual model ID and API errors. Do not fabricate live
    results.
-   Acceptance: live mode is distinct from offline mode and fails
    clearly when unavailable.

### Phase 8 --- Dashboard and presentation polish

-   Only after CLI and tests pass, add a simple Streamlit dashboard if
    time allows.
-   Include task/seed/policy controls, run trace, metric comparison,
    chart, and download of logs/results.
-   Do not make the dashboard a dependency for core execution.
-   Acceptance: panel can run the prevalidated scenario and inspect how
    it worked.

### Phase 9 --- Research extensions

-   Add pilot-test baseline, offline prefix-branching comparison,
    feature ablation, and cross-model transfer only after MVP passes.
-   Update the status table in README and progress log. Be explicit
    about what is incomplete.

After every phase, update `PROGRESS_LOG.md` with exact files/functions
changed, commands executed, actual pass/fail output, known issues,
completed/in-progress/not-started items, and the next exact action. If a
session is interrupted, the next session must read and summarize the log
before continuing.

------------------------------------------------------------------------

## 11. Automated acceptance tests

At minimum, implement tests for: 1. Task definitions load and validate.
2. The same seed produces the same injected fault and scenario hash. 3.
Different strategies receive equivalent scenario configurations. 4. The
Coder prompt does not contain the original full task prompt. 5. Critic
probabilities are finite and in `[0, 1]`; invalid output is handled
safely. 6. Features are computed only from information available at that
step. 7. `no_intervention` always continues and ignores risk for its
decision. 8. `naive_threshold` aborts iff `failure_prob > 0.5`, as
specified. 9. Proposed inference loads the trained artifact and makes a
single classifier decision. 10. Proposed live mode never invokes offline
branching or rollback. 11. Intervention appends corrective guidance and
replans from the current step. 12. A verifier failure cannot be counted
as success. 13. Metrics match hand-calculated small test fixtures. 14.
JSONL logs are valid and contain required identifiers/fields. 15.
Offline mode works with no `.env` and no network. 16. Missing/invalid
API key and provider errors produce clear errors, not fake success. 17.
Aggregate metrics can be recomputed from saved logs. 18. A clean setup
from README commands succeeds in a fresh virtual environment.

Run `pytest` after each meaningful change and record actual output. Do
not claim tests passed unless they were run.

------------------------------------------------------------------------

## 12. Demo command and output contract

Aim for a simple interface similar to:

``` bash
python run_demo.py --task task_03 --seed 42 --mode offline
python run_demo.py --task task_03 --seed 42 --mode live
python run_demo.py --suite --seed 42 --mode offline
python policy/train_policy.py
python visualize_trajectory.py --run-id <actual-run-id>
```

Adapt commands to the actual implementation and document exact commands
in README. Do not make a command claim to run all modes if it runs only
one.

For a single-task demo, print each policy's risk estimates, features,
decisions, intervention events, verifier outcome, and final result. For
suite runs, show aggregate metrics with denominators and the number of
tasks/seeds. Produce a chart from actual run logs. Never copy the
illustrative success-rate table from the original brief into runtime
output.

------------------------------------------------------------------------

## 13. Academic integrity and research defensibility

-   The proposed policy is not guaranteed to beat baselines. Test it and
    report what happens.
-   The synthetic task suite demonstrates mechanics; it does not alone
    validate generalization to production systems.
-   An LLM-generated risk estimate is not a calibrated probability
    unless calibration is evaluated.
-   A heuristic reversibility feature is a proxy, not a proven measure.
-   A small synthetic classifier may overfit. Report the size, split
    protocol, class balance, and limitations.
-   A reproduced baseline must be methodologically faithful; otherwise
    label it an approximation.
-   Do not claim the 20-paper literature review or cited paper results
    have been independently verified merely because they appear in the
    supplied report.
-   Preserve the four research objectives in the documentation:
    -   **C1:** rollout-free versus counterfactual/prefix-branching
        controller.
    -   **C2:** feature ablation.
    -   **C3:** cost-sensitive decision framework.
    -   **C4:** cross-model generalization.
-   The MVP primarily establishes a working testbed and initial
    evidence; it does not automatically complete C1--C4.
-   The optional human-deferral action is out of scope until the
    committed objectives and demo work.

------------------------------------------------------------------------

## 14. What Codex must do first

Start by: 1. Listing the working-folder/repository files and checking whether there is already code. 2. Treating this document as the complete and only required specification; do not require the original brief, report, or presentation. 3. Reporting any conflicts or implementation risks found in the repository or this document. 4. Proposing a short phased plan with concrete acceptance tests. 5. Creating `PROGRESS_LOG.md`. 6. Implementing **Phase 1 only**: the offline deterministic vertical slice, including tests. 7. Running the tests and reporting exact commands and actual results. 8. Stopping at the Phase 1 checkpoint and stating the next precise step.

Do not start with the dashboard, cloud deployment, full external
benchmarks, or broad live API experimentation. Do not add unnecessary
abstractions. Prefer a small working system that can be expanded over a
large incomplete codebase.

**Priority order:** correctness → reproducibility → working offline demo
→ honest metrics → live model integration → research extensions →
dashboard polish.
