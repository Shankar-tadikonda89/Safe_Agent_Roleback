"""Interactive, offline-first academic dashboard for the capstone."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise SystemExit(
        "Streamlit is not installed. From the project root, install the optional "
        "dashboard dependencies with `python -m pip install -r requirements-dashboard.txt`."
    ) from exc

import pandas as pd

from dashboard.helpers import (
    PHASE98_POLICY_LABELS,
    PHASE98_RESULTS_DIR,
    PROJECT_ROOT,
    RESEARCH_LIMITATION,
    TASKS,
    downloadable_artifacts,
    load_phase98_pilot_results,
    planner_output_for_task,
    run_offline_scenario,
    temporary_trace_path,
    trace_table_rows,
    trajectory_chart_rows,
)
from experiments.metrics import POLICIES


PAGE_NAMES = (
    "Project Overview",
    "Interactive Execution",
    "Decision Explanation",
    "No-Rollback Demonstration",
    "Research Results",
)
POLICY_LABELS = {
    "no_intervention": "No intervention",
    "naive_threshold": "Naive threshold",
    "proposed": "Proposed (Phase 4 binary model)",
}


st.set_page_config(
    page_title="Safe Intervention Without Rollback",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #F3F6FB;
      }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stMain"] h1, [data-testid="stMain"] h2,
      [data-testid="stMain"] h3, [data-testid="stMain"] h4,
      [data-testid="stMain"] p, [data-testid="stMain"] label,
      [data-testid="stMain"] [data-testid="stMetricLabel"],
      [data-testid="stMain"] [data-testid="stMetricValue"] {
        color: #172B4D;
      }
      [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
      [data-testid="stMain"] .stMarkdown small { color: #52647A; }
      [data-testid="stSidebar"] { background: #10233F; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] label { color: #FFFFFF; }
      .hero { padding: clamp(1rem, 3vw, 1.6rem); border-radius: 16px;
        color: #FFFFFF; background: linear-gradient(120deg,#10233F,#0F766E);
        margin-bottom: 1.15rem; }
      .hero h1 { color: #FFFFFF !important; margin: 0 0 .4rem 0; font-size: clamp(1.65rem, 3vw, 2.25rem); }
      .hero p { color: #F2F7FF !important; margin: 0; line-height: 1.55; }
      .note { padding: .9rem 1rem; border-left: 4px solid #D97706;
        border-radius: 8px; background: #FFF7E6; color: #92400E; margin: .5rem 0 1rem; }
      .info-card { border: 1px solid #DCE4EF; border-radius: 14px; padding: 1rem 1.1rem;
        background: #FFFFFF; min-height: 175px; box-shadow: 0 2px 8px #172B4D0A; }
      .info-card h3 { margin-top: 0; color: #172B4D; }
      .info-card p { color: #52647A; line-height: 1.55; }
      .flow { display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: .45rem; padding: .8rem; border: 1px solid #DCE4EF;
        border-radius: 12px; background: #FFFFFF; color: #172B4D; font-weight: 650; }
      .flow span { background: #EFF6FF; border-radius: 8px; padding: .6rem .7rem; }
      [data-testid="stMain"] button[kind="primary"] {
        background: #2563EB; color: #FFFFFF; border: 1px solid #1D4ED8;
        font-weight: 650; min-height: 2.7rem;
      }
      [data-testid="stMain"] button[kind="primary"]:hover { background: #1D4ED8; border-color: #1E40AF; }
      [data-testid="stMain"] button:focus-visible { outline: 3px solid #0F766E; outline-offset: 2px; }
      [data-testid="stMain"] [data-testid="stAlert"] { border-radius: 10px; }
      [data-testid="stMain"] [data-testid="stExpander"] { border-color: #DCE4EF; border-radius: 10px; }
      [data-testid="stMain"] [data-testid="stDataFrame"] { border: 1px solid #DCE4EF; border-radius: 10px; }
      @media (max-width: 700px) {
        .info-card { min-height: auto; }
        .flow { justify-content: flex-start; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🧭 Safe Intervention")
    st.caption("Offline academic demonstration")
    page = st.radio("Navigate", PAGE_NAMES, label_visibility="collapsed")
    st.divider()
    st.caption("No Gemini/provider calls are made by this dashboard.")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def current_result() -> dict[str, Any] | None:
    result = st.session_state.get("dashboard_last_result")
    return result if isinstance(result, dict) else None


def decision_reason(policy: str, event: dict[str, Any]) -> str:
    risk = event.get("failure_prob")
    action = event.get("decision")
    if policy == "no_intervention":
        return "This baseline always continues by definition."
    if policy == "naive_threshold":
        return f"Abort only when the input risk score is strictly greater than 0.5; this run used {risk!r}."
    details = event.get("policy_decision_details") or {}
    helpful = details.get("helpful_intervention_probability")
    return (
        f"The Phase 4 binary model selected {action!r} from its predicted helpful-intervention "
        f"probability ({helpful!r}) and its own expected costs. The feature `failure_prob` input "
        f"was {risk!r}; it is not the helpfulness prediction."
    )


if page == "Project Overview":
    hero(
        "Safe Intervention Without Rollback",
        "A Lightweight, Rollout-Free Decision Policy for Proactive Failure Prevention in LLM Multi-Agent Systems.",
    )
    st.warning(RESEARCH_LIMITATION, icon="⚠️")
    st.subheader("Research objective")
    st.write(
        "Study whether a lightweight decision policy can identify when corrective guidance may help "
        "an agent continue, while preserving its execution history instead of restoring an earlier state."
    )

    st.markdown("### Problem, approach, and safety principle")
    overview_cards = st.columns(3)
    card_copy = (
        ("The problem", "A faulty instruction or plan can send a multi-agent workflow toward an incorrect result. The controller must decide from evidence available at that decision point."),
        ("The approach", "The offline application extracts trajectory features and delegates action selection to the existing policy. Applied repair adds corrective guidance and replans from the current state."),
        ("No rollback", "Prior code and events remain in the execution history. The Phase 9.8 pilot is a separate counterfactual simulator; its selected interventions were not applied."),
    )
    for col, (title, body) in zip(overview_cards, card_copy):
        with col:
            st.markdown(f'<section class="info-card"><h3>{title}</h3><p>{body}</p></section>', unsafe_allow_html=True)

    st.markdown("### Offline application architecture")
    st.markdown(
        '<div class="flow" aria-label="Planner to feature extraction to policy to coder to verifier">'
        '<span>Planner</span><b aria-hidden="true">→</b><span>Feature extraction</span><b aria-hidden="true">→</b>'
        '<span>Decision policy</span><b aria-hidden="true">→</b><span>Coder</span><b aria-hidden="true">→</b><span>Verifier</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Deterministic local components; no live LLM run. The feature and policy calls reuse the existing application code.")
    st.info(
        "Keep the evidence types distinct: the Interactive Execution page runs the Phase 4 arithmetic application; "
        "Research Results displays the frozen Phase 9.8 synthetic pilot."
    )

elif page == "Interactive Execution":
    hero("Interactive Execution", "Run one real task through the existing offline application.")
    st.warning(
        "Offline only. The Proposed option here is the older Phase 4 binary helpfulness model. "
        "It is not the Phase 9.8 four-class pilot policy.",
        icon="⚠️",
    )
    task_by_id = {task.task_id: task for task in TASKS}
    with st.form("offline_execution_form", clear_on_submit=False):
        col_a, col_b, col_c = st.columns([1.3, 1, 1.4])
        with col_a:
            task_id = st.selectbox(
                "Task", list(task_by_id),
                format_func=lambda key: f"{key} · {task_by_id[key].title}",
            )
        with col_b:
            seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
        with col_c:
            policy = st.selectbox(
                "Policy", POLICIES,
                format_func=lambda name: POLICY_LABELS.get(name, name),
            )
        run_clicked = st.form_submit_button("Run offline demonstration", type="primary")

    if run_clicked:
        try:
            if "dashboard_session_id" not in st.session_state:
                import uuid
                st.session_state["dashboard_session_id"] = uuid.uuid4().hex
            trace_path = temporary_trace_path(
                st.session_state["dashboard_session_id"], task_id, int(seed), policy,
            )
            with st.spinner("Running the deterministic Planner → policy → Coder → Verifier workflow…"):
                result = run_offline_scenario(task_id, int(seed), policy, trace_path=trace_path)
            result["planner_output"] = planner_output_for_task(task_id)
            result["trace_path"] = str(trace_path)
            st.session_state["dashboard_last_result"] = result
            st.session_state["dashboard_last_inputs"] = {
                "task_id": task_id, "seed": int(seed), "policy": policy,
            }
            st.progress(1.0, text="Offline execution completed")
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.session_state.pop("dashboard_last_result", None)
            st.error(f"The offline run could not complete: {exc}")
        except Exception:
            st.session_state.pop("dashboard_last_result", None)
            st.error("The offline run failed unexpectedly. The CLI remains available for diagnosis.")

    result = current_result()
    if result:
        summary = result["summary"]
        events = result.get("events", [])
        st.divider()
        st.subheader(f"Run trace · {summary['task_id']} · seed {summary['seed']}")
        info = st.columns(4)
        info[0].metric("Final oracle result", "PASS" if summary["passed"] else "FAIL")
        info[1].metric("Policy", POLICY_LABELS.get(summary["policy"], summary["policy"]))
        info[2].metric("Executed steps", summary["steps_executed"])
        info[3].metric("Applied interventions", summary["intervention_count"])
        st.caption(
            f"Scenario `{summary['scenario_hash']}` · injected fault: {summary['fault_type']} · "
            f"aborted: {summary['aborted']} · offline LLM calls: {summary['llm_calls']}"
        )

        with st.expander("Planner output", expanded=True):
            for instruction in result.get("planner_output", []):
                st.code(instruction, language="text")
        if not events:
            st.info("The run produced no step events.")
        else:
            event = events[-1]
            st.markdown("### Decision-time observation and policy output")
            st.dataframe(trace_table_rows(events), hide_index=True, use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.markdown("**Feature values passed to the policy**")
                st.json(event.get("features", {}))
                st.caption("`failure_prob` is the risk input. It is not the Phase 4 model's helpfulness prediction.")
            with right:
                details = event.get("policy_decision_details")
                st.markdown("**Policy prediction and selected action**")
                if details:
                    st.json(details)
                else:
                    st.write(f"Selected action: **{event.get('decision')}**")
                    st.write(event.get("decision_reason", ""))
                if event.get("decision") == "abort":
                    st.warning("The policy aborted before executing this step; no verifier result exists for it.")
            st.markdown("### Coder output")
            st.code(event.get("code", "") or "(no code produced)", language="python")
            st.markdown("### Verifier result")
            if event.get("verifier") is not None:
                st.json(event["verifier"])
            else:
                st.info("Verifier did not run because the policy aborted before execution.")
            st.metric("Final result", "PASS" if summary["passed"] else "FAIL")

        trace_path = Path(result.get("trace_path", ""))
        if trace_path.is_file():
            st.download_button(
                "Download this run's temporary JSONL trace",
                data=trace_path.read_bytes(),
                file_name=trace_path.name,
                mime="application/x-ndjson",
                key="download_current_temp_trace",
            )
            st.caption("The trace is stored in a unique OS temporary directory, separate from frozen results.")

elif page == "Decision Explanation":
    hero("Decision Explanation", "Inspect the exact features and rule used by the selected policy.")
    result = current_result()
    if not result or not result.get("events"):
        st.info("Run a task on Interactive Execution to see its actual decision record here.")
    else:
        event = result["events"][-1]
        policy = result["summary"]["policy"]
        st.markdown(f"### {POLICY_LABELS.get(policy, policy)}")
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Input features**")
            st.json(event.get("features", {}))
            st.metric("Failure risk input", f"{event.get('failure_prob', float('nan')):.2f}")
            st.caption("The offline arithmetic suite supplies a neutral 0.5 prior; it is not calibrated failure probability.")
        with right:
            st.markdown("**Observed decision**")
            st.metric("Selected action", str(event.get("decision", "unknown")).upper())
            st.write(decision_reason(policy, event))
            details = event.get("policy_decision_details")
            if policy == "proposed" and details:
                st.markdown("**Phase 4 binary model prediction and costs**")
                st.json(details)
                st.latex(r"E[C_{continue}] = 3p_{helpful}")
                st.latex(r"E[C_{intervene}] = 2 - p_{helpful}")
                st.caption("These binary helpfulness costs belong to the Phase 4 CLI model.")
            elif policy == "naive_threshold":
                st.latex(r"\text{abort iff failure\_prob} > 0.5")
                st.caption("An abort is a failed task, not a repair intervention.")
            else:
                st.info("The no-intervention baseline always continues.")

    with st.expander("Phase 9.8 research policy rule (separate implementation)"):
        st.markdown(
            "This is the four-class cost-sensitive model used in the frozen Phase 9.8 pilot. "
            "It is not the older Phase 4 binary model used by Interactive Execution."
        )
        st.latex(r"E[C_{continue}] = 3(q_R + q_N)")
        st.latex(r"E[C_{intervene}] = 2 + 3(q_N + q_H)")
        st.write("Select INTERVENE only when its expected cost is strictly lower; ties select CONTINUE.")

elif page == "No-Rollback Demonstration":
    hero("No-Rollback Demonstration", "Follow the actual execution records from a recent offline run.")
    result = current_result()
    if not result:
        st.info("Run a task on Interactive Execution to populate this view with real execution records.")
    else:
        summary = result["summary"]
        events = result.get("events", [])
        st.warning(
            "This view shows the selected task's executed offline trace. It does not display counterfactual outcomes "
            "as applied actions.",
            icon="⚠️",
        )
        st.markdown("#### 1 · Initial plan")
        for instruction in result.get("planner_output", []):
            st.code(instruction, language="text")
        for event in events:
            with st.container(border=True):
                st.markdown(f"#### Step {event.get('step_index', '?')} · decision point")
                st.write("**Observed instruction:**", event.get("faulted_instruction", "(not recorded)"))
                st.write("**Decision:**", event.get("decision"), "· risk input:", event.get("failure_prob"))
                if event.get("decision") == "abort":
                    st.warning("Execution stopped before this step. The verifier did not run.")
                elif event.get("intervention"):
                    st.success("An intervention was applied in this offline execution path.")
                    st.write("**Corrective guidance:**", event.get("corrective_guidance"))
                    st.write("**Planner replanned:**", bool(event.get("planner_replanned")))
                    st.write("**Code preserved before this step:**")
                    st.code(event.get("code_before", ""), language="python")
                else:
                    st.info("No intervention was applied at this decision.")
                st.write("**Coder output after this step:**")
                st.code(event.get("code", "") or "(none)", language="python")
                if event.get("verifier") is not None:
                    st.write("**Verifier feedback:**")
                    st.json(event["verifier"])
        st.markdown("#### Final verification")
        st.metric("Final oracle result", "PASS" if summary["passed"] else "FAIL")
        st.write(
            f"Applied interventions: {summary['intervention_count']} · "
            f"final result: {summary['final_result']!r} · aborted: {summary['aborted']}"
        )
        st.caption("History is retained; the offline repair path appends guidance and continues from current code.")

else:  # Research Results
    hero("Research Results", "The recorded Phase 9.8 pilot is shown separately from the interactive application.")
    st.warning(
        "Synthetic simulator study only. These are not live LLM results or a real-world effectiveness trial. "
        "Selected counterfactual actions were not applied.",
        icon="⚠️",
    )
    results, error = load_phase98_pilot_results(PROJECT_ROOT)
    if error:
        st.error(f"Frozen Phase 9.8 results are unavailable: {error}")
    else:
        metrics = results["policy_metrics"]
        primary = results["primary_comparison"]
        cards = st.columns(4)
        cards[0].metric("Scenarios", f"{results['scenario_count']:,}")
        cards[1].metric("Matched decisions", f"{results['matched_decision_count']:,}")
        cards[2].metric("Proposed interventions", metrics["proposed"]["intervention_count"])
        cards[3].metric("Prefix selections", metrics["phase9_8_native_prefix_forecast_v1"]["intervention_count"])

        table_rows = []
        for policy in PHASE98_POLICY_LABELS:
            item = metrics[policy]
            table_rows.append({
                "Policy": PHASE98_POLICY_LABELS[policy],
                "Success rate": f"{item['task_success_rate']:.2%}",
                "Successes / scenarios": f"{item['success_count']:,} / {item['attempts']:,}",
                "Mean cost": item["mean_realized_task_cost"],
                "Selected interventions": item["intervention_count"],
                "Confirmed applications": item["confirmed_application_count"],
            })
        st.markdown("### Matched comparative policies")
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
        chart_data = pd.DataFrame(
            {
                "Success rate": [metrics[name]["task_success_rate"] for name in PHASE98_POLICY_LABELS],
                "Mean cost": [metrics[name]["mean_realized_task_cost"] for name in PHASE98_POLICY_LABELS],
            },
            index=[PHASE98_POLICY_LABELS[name] for name in PHASE98_POLICY_LABELS],
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Task success rate**")
            st.bar_chart(chart_data[["Success rate"]], horizontal=True, sort=False, color="#2563EB")
        with col_b:
            st.markdown("**Mean synthetic cost**")
            st.bar_chart(chart_data[["Mean cost"]], horizontal=True, sort=False, color="#0F766E")

        st.markdown("### Primary paired cost comparison")
        primary_cols = st.columns(3)
        primary_cols[0].metric("Proposed − no-intervention", f"{primary['mean_cost_difference']:.1f}")
        primary_cols[1].metric(
            "95% interval", f"[{primary['interval_95'][0]:.1f}, {primary['interval_95'][1]:.1f}]",
        )
        primary_cols[2].metric("Bootstrap resamples", f"{primary['bootstrap_resamples']:,}")
        st.caption(
            "Primary estimate and interval are transcribed from the completed post-pilot audit; they are not fields "
            "in comparative_metrics.json. The proposed policy selected continue in every scenario, so its paired "
            "cost difference is zero in this pilot. This is not evidence of equivalence or superiority."
        )

        st.markdown("### Interpretation and limits")
        st.write(
            f"The proposed policy selected **{metrics['proposed']['intervention_count']}** interventions and tied "
            f"the no-intervention baseline. The prefix comparator selected **{metrics['phase9_8_native_prefix_forecast_v1']['intervention_count']} counterfactual interventions**; "
            "the pilot records zero confirmed applications. Those selections were not applied to an agent."
        )
        st.write(
            "The `pilot_test` action is outcome-selected descriptive information and is excluded from comparative metrics. "
            f"It selected **{results['pilot_test']['selected_action']}** using {results['pilot_test']['scenarios_used']:,} "
            "pilot scenarios."
        )
        st.info(
            "The pilot covers ten configuration groups, is synthetic and exploratory, and does not establish real-world "
            "critic calibration or policy effectiveness. The prefix comparator has privileged simulator information "
            "and compute matching/deployability have not been established."
        )
        with st.expander("Metric definitions and provenance"):
            st.write("Success rate = verifier-passing scenarios / attempts (2,560 per policy). Mean cost is per scenario.")
            st.write("Intervention count is selected counterfactual actions; confirmed applications remain zero.")
            st.write(f"Metrics JSON: `{results['metrics_path']}`")
            st.write(f"Scored JSONL: `{results['scored_runs_path']}`")
            st.write(f"Primary estimate source: `{primary['source']}`")
            st.write("Latency in these records is local simulation/runtime latency, not Gemini/API latency.")

        for relative in (results["metrics_path"], results["scored_runs_path"], results["pilot_test_path"]):
            path = PROJECT_ROOT / relative
            if path.is_file():
                st.download_button(
                    f"Download {path.name}", path.read_bytes(), file_name=path.name,
                    mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    key=f"download_{path.name}",
                )

        with st.expander("Earlier Phase 6 held-out task evaluation (different experiment)"):
            st.caption("Separate arithmetic-task results; do not combine these values with the Phase 9.8 pilot.")
            heldout_metrics_path = PROJECT_ROOT / "results/heldout_evaluation/comparison_metrics.csv"
            if heldout_metrics_path.is_file():
                st.download_button(
                    "Download Phase 6 metrics CSV", heldout_metrics_path.read_bytes(),
                    file_name=heldout_metrics_path.name, mime="text/csv", key="download_phase6_csv",
                )
            else:
                st.info("No Phase 6 metrics CSV is available at the documented path.")

    st.markdown("### Existing offline artifacts")
    available = downloadable_artifacts(PROJECT_ROOT)
    if available:
        selected = st.selectbox(
            "Offline log or result file", available,
            format_func=lambda path: str(path.relative_to(PROJECT_ROOT)),
        )
        if selected.is_file():
            st.download_button(
                "Download selected offline artifact", selected.read_bytes(),
                file_name=selected.name,
                mime=mimetypes.guess_type(selected.name)[0] or "application/octet-stream",
                key="download_existing_offline_artifact",
            )
    else:
        st.info("No allowlisted offline artifacts are available.")

st.sidebar.caption("Research demo · synthetic data · no live provider requests")
