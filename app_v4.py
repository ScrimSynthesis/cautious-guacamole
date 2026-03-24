import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Batch Behavior Explorer",
    page_icon="🧪",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def gaussian_score(x: float, center: float, sigma: float) -> float:
    """Returns a 0-100 score with 100 at center."""
    return 100.0 * math.exp(-0.5 * ((x - center) / sigma) ** 2)


@dataclass
class Scenario:
    temp_c: float
    ph: float
    agitation_rpm: float
    feed_rate: float
    hold_time_min: float
    solids_pct: float
    ratio: float


def evaluate_scenario(s: Scenario) -> dict:
    """
    Educational heuristic model.
    This is not a plant-validated predictive model.
    It is meant to help learn variable interactions and operating tradeoffs.
    """

    # Base variable scores
    temp_score = gaussian_score(s.temp_c, 82, 8)
    ph_score = gaussian_score(s.ph, 5.8, 0.45)
    agitation_score = gaussian_score(s.agitation_rpm, 240, 55)
    feed_score = gaussian_score(s.feed_rate, 1.20, 0.30)
    hold_score = gaussian_score(s.hold_time_min, 95, 20)
    solids_score = gaussian_score(s.solids_pct, 32, 6)
    ratio_score = gaussian_score(s.ratio, 1.05, 0.10)

    base_yield = (
        0.24 * temp_score
        + 0.22 * ph_score
        + 0.10 * agitation_score
        + 0.09 * feed_score
        + 0.14 * hold_score
        + 0.08 * solids_score
        + 0.13 * ratio_score
    )

    base_consistency = (
        0.18 * temp_score
        + 0.20 * ph_score
        + 0.18 * agitation_score
        + 0.12 * feed_score
        + 0.12 * hold_score
        + 0.10 * solids_score
        + 0.10 * ratio_score
    )

    risk = 12.0
    notes = []
    penalties = []

    def add_effect(text: str, yield_pen=0, cons_pen=0, risk_add=0):
        penalties.append((text, yield_pen, cons_pen, risk_add))

    # Interactions / operating logic
    if s.temp_c > 90 and s.hold_time_min > 110:
        add_effect(
            "High temperature + long hold may increase degradation risk.",
            yield_pen=10,
            cons_pen=8,
            risk_add=18,
        )
        notes.append("Possible overprocessing from hot + long hold.")

    if s.temp_c < 74:
        add_effect(
            "Lower temperature may slow conversion and extend cycle time.",
            yield_pen=8,
            cons_pen=4,
            risk_add=8,
        )
        notes.append("Reaction may be underdriven at lower temperature.")

    if s.ph < 5.2 and s.temp_c > 88:
        add_effect(
            "Low pH combined with high temperature can increase instability.",
            yield_pen=7,
            cons_pen=10,
            risk_add=16,
        )
        notes.append("This combo may widen batch-to-batch variability.")

    if s.agitation_rpm < 170 and s.solids_pct > 36:
        add_effect(
            "High solids with lower agitation can hurt mixing quality.",
            yield_pen=6,
            cons_pen=12,
            risk_add=15,
        )
        notes.append("Watch for poor suspension or localized concentration effects.")

    if s.feed_rate > 1.45 and s.agitation_rpm < 200:
        add_effect(
            "Faster feed with modest mixing can raise local upset risk.",
            yield_pen=8,
            cons_pen=9,
            risk_add=14,
        )
        notes.append("The system may not absorb feed changes smoothly.")

    if abs(s.ratio - 1.05) > 0.15:
        add_effect(
            "Stoichiometric ratio is far from the target zone.",
            yield_pen=11,
            cons_pen=7,
            risk_add=12,
        )
        notes.append("Composition balance may be pushing off-target behavior.")

    if s.ph > 6.4:
        add_effect(
            "Higher pH may move the process away from the preferred operating window.",
            yield_pen=7,
            cons_pen=5,
            risk_add=8,
        )
        notes.append("pH is on the high side of the preferred region.")

    if s.hold_time_min < 65:
        add_effect(
            "Short hold time may reduce completion margin.",
            yield_pen=9,
            cons_pen=6,
            risk_add=10,
        )
        notes.append("Batch may not have enough time to settle into target behavior.")

    # Apply penalties
    yield_score = base_yield
    consistency_score = base_consistency

    for _, y_pen, c_pen, r_add in penalties:
        yield_score -= y_pen
        consistency_score -= c_pen
        risk += r_add

    # Additional risk from distance away from nominal
    risk += 0.10 * (100 - temp_score)
    risk += 0.12 * (100 - ph_score)
    risk += 0.10 * (100 - agitation_score)
    risk += 0.08 * (100 - hold_score)
    risk += 0.06 * (100 - feed_score)
    risk += 0.06 * (100 - solids_score)
    risk += 0.06 * (100 - ratio_score)

    yield_score = clamp(yield_score, 0, 100)
    consistency_score = clamp(consistency_score, 0, 100)
    risk = clamp(risk, 0, 100)

    # Estimated cycle time (educational approximation)
    cycle_time_hr = 6.2
    cycle_time_hr += max(0, (80 - s.temp_c) * 0.05)
    cycle_time_hr += max(0, (95 - s.hold_time_min) * -0.01)  # shorter hold shortens time slightly
    cycle_time_hr += max(0, (s.hold_time_min - 95) * 0.02)
    cycle_time_hr += max(0, (1.10 - s.feed_rate) * 1.1)
    cycle_time_hr += max(0, (34 - s.agitation_rpm) * 0.0)  # no effect, placeholder for clarity
    cycle_time_hr += max(0, (s.solids_pct - 35) * 0.05)
    cycle_time_hr = clamp(cycle_time_hr, 4.5, 11.5)

    # Spec probability
    spec_probability = clamp(
        0.45 * yield_score + 0.35 * consistency_score + 0.20 * (100 - risk),
        0,
        100,
    )

    # Overall health
    overall = clamp(
        0.40 * yield_score + 0.35 * consistency_score + 0.25 * (100 - risk),
        0,
        100,
    )

    # Primary driver ranking
    driver_scores = {
        "Temperature": temp_score,
        "pH": ph_score,
        "Agitation": agitation_score,
        "Feed Rate": feed_score,
        "Hold Time": hold_score,
        "Solids": solids_score,
        "Ratio": ratio_score,
    }
    sensitivity_order = sorted(driver_scores.items(), key=lambda x: x[1])

    if not notes:
        notes.append("This scenario sits fairly close to the preferred operating zone.")

    return {
        "yield_score": round(yield_score, 1),
        "consistency_score": round(consistency_score, 1),
        "risk_score": round(risk, 1),
        "cycle_time_hr": round(cycle_time_hr, 2),
        "spec_probability": round(spec_probability, 1),
        "overall_score": round(overall, 1),
        "driver_scores": driver_scores,
        "weakest_drivers": sensitivity_order[:3],
        "notes": notes,
        "penalties": penalties,
    }


def scenario_inputs(prefix: str, defaults: dict) -> Scenario:
    return Scenario(
        temp_c=st.slider(
            "Temperature (°C)",
            min_value=60.0,
            max_value=105.0,
            value=float(defaults["temp_c"]),
            step=1.0,
            key=f"{prefix}_temp_c",
        ),
        ph=st.slider(
            "pH",
            min_value=4.5,
            max_value=7.0,
            value=float(defaults["ph"]),
            step=0.05,
            key=f"{prefix}_ph",
        ),
        agitation_rpm=st.slider(
            "Agitation (rpm)",
            min_value=80,
            max_value=400,
            value=int(defaults["agitation_rpm"]),
            step=5,
            key=f"{prefix}_agitation_rpm",
        ),
        feed_rate=st.slider(
            "Feed Rate (relative units)",
            min_value=0.60,
            max_value=2.00,
            value=float(defaults["feed_rate"]),
            step=0.05,
            key=f"{prefix}_feed_rate",
        ),
        hold_time_min=st.slider(
            "Hold Time (min)",
            min_value=30,
            max_value=180,
            value=int(defaults["hold_time_min"]),
            step=5,
            key=f"{prefix}_hold_time_min",
        ),
        solids_pct=st.slider(
            "Solids (%)",
            min_value=10.0,
            max_value=50.0,
            value=float(defaults["solids_pct"]),
            step=1.0,
            key=f"{prefix}_solids_pct",
        ),
        ratio=st.slider(
            "Stoichiometric Ratio",
            min_value=0.70,
            max_value=1.40,
            value=float(defaults["ratio"]),
            step=0.01,
            key=f"{prefix}_ratio",
        ),
    )


def metric_card_row(result: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Yield Score", f"{result['yield_score']}/100")
    c2.metric("Consistency Score", f"{result['consistency_score']}/100")
    c3.metric("Risk Score", f"{result['risk_score']}/100")
    c4.metric("Cycle Time", f"{result['cycle_time_hr']} hr")
    c5.metric("In-Spec Probability", f"{result['spec_probability']}%")


def make_radar(result: dict) -> go.Figure:
    labels = list(result["driver_scores"].keys())
    values = list(result["driver_scores"].values())
    values = values + values[:1]
    labels = labels + labels[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="Scenario",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=450,
    )
    return fig


def make_profile_plot(s: Scenario) -> go.Figure:
    time = np.arange(0, 181, 5)

    # Simulated temperature profile
    ramp_target = s.temp_c
    temp_profile = 25 + (ramp_target - 25) * (1 - np.exp(-time / 35))
    temp_profile = np.where(time > 75, ramp_target + 0.6 * np.sin(time / 18), temp_profile)

    # Simulated pH behavior
    ph_profile = 6.5 - (6.5 - s.ph) * (1 - np.exp(-time / 22))
    ph_profile = np.where(time > 55, s.ph + 0.06 * np.sin(time / 15), ph_profile)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=temp_profile, mode="lines", name="Temperature (°C)"))
    fig.add_trace(go.Scatter(x=time, y=ph_profile, mode="lines", name="pH", yaxis="y2"))

    fig.update_layout(
        title="Illustrative Batch Profile",
        xaxis=dict(title="Time (min)"),
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(
            title="pH",
            overlaying="y",
            side="right",
            range=[4.0, 7.2],
        ),
        legend=dict(orientation="h", y=1.1),
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_safe_zone_map(base: Scenario) -> go.Figure:
    temps = np.linspace(60, 105, 40)
    phs = np.linspace(4.5, 7.0, 40)

    z = np.zeros((len(phs), len(temps)))

    for i, ph in enumerate(phs):
        for j, temp in enumerate(temps):
            test = Scenario(
                temp_c=float(temp),
                ph=float(ph),
                agitation_rpm=base.agitation_rpm,
                feed_rate=base.feed_rate,
                hold_time_min=base.hold_time_min,
                solids_pct=base.solids_pct,
                ratio=base.ratio,
            )
            result = evaluate_scenario(test)
            # High value = healthier
            z[i, j] = result["overall_score"]

    fig = go.Figure(
        data=go.Heatmap(
            x=temps,
            y=phs,
            z=z,
            colorbar=dict(title="Health"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[base.temp_c],
            y=[base.ph],
            mode="markers",
            marker=dict(size=12, symbol="x"),
            name="Current Scenario",
        )
    )

    fig.update_layout(
        title="Safe Operating Zone Map (Temp vs pH)",
        xaxis_title="Temperature (°C)",
        yaxis_title="pH",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def compare_dataframe(result_a: dict, result_b: dict) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Metric": [
                "Yield Score",
                "Consistency Score",
                "Risk Score",
                "Cycle Time (hr)",
                "In-Spec Probability",
                "Overall Score",
            ],
            "Run A": [
                result_a["yield_score"],
                result_a["consistency_score"],
                result_a["risk_score"],
                result_a["cycle_time_hr"],
                result_a["spec_probability"],
                result_a["overall_score"],
            ],
            "Run B": [
                result_b["yield_score"],
                result_b["consistency_score"],
                result_b["risk_score"],
                result_b["cycle_time_hr"],
                result_b["spec_probability"],
                result_b["overall_score"],
            ],
        }
    )
    df["Delta (B - A)"] = df["Run B"] - df["Run A"]
    return df


VARIABLE_GUIDE = {
    "Temperature": {
        "what": "Controls reaction energy and often influences speed, completion, and side-reaction risk.",
        "why": "A strong driver of conversion and stability in many batch systems.",
        "low": "Too low can mean slow reaction, incomplete conversion, and longer cycle times.",
        "high": "Too high can increase degradation, overshoot, or variability.",
        "interacts": "Strongly interacts with hold time and pH.",
    },
    "pH": {
        "what": "Represents acidity/basicity and can strongly influence reaction pathway and stability.",
        "why": "Often one of the most sensitive process variables in liquid chemistry.",
        "low": "Too low can increase instability or move the system out of the preferred chemistry zone.",
        "high": "Too high may reduce performance or shift the product away from target behavior.",
        "interacts": "Strongly interacts with temperature and ratio.",
    },
    "Agitation": {
        "what": "Controls mixing, solids suspension, and how evenly the system behaves.",
        "why": "Poor mixing can create local hot spots, concentration gradients, and inconsistency.",
        "low": "Too low can reduce uniformity, especially at high solids.",
        "high": "Too high may add shear or inefficiency without meaningful benefit.",
        "interacts": "Strongly interacts with solids and feed rate.",
    },
    "Feed Rate": {
        "what": "Controls how fast reactants or additives enter the system.",
        "why": "Too fast can shock the system, too slow can reduce throughput.",
        "low": "Can stretch cycle time and underutilize the reactor.",
        "high": "Can create localized upset if mixing and chemistry cannot keep up.",
        "interacts": "Strongly interacts with agitation.",
    },
    "Hold Time": {
        "what": "How long the batch stays in the reaction or conditioning window.",
        "why": "It affects completion, maturity, and total cycle time.",
        "low": "Too short can reduce completion margin.",
        "high": "Too long can hurt throughput and sometimes raise degradation risk.",
        "interacts": "Strongly interacts with temperature.",
    },
    "Solids": {
        "what": "Represents concentration and slurry heaviness.",
        "why": "Higher solids can change mixing behavior, transfer, and consistency.",
        "low": "May reduce process intensity or desired concentration.",
        "high": "Can make mixing harder and raise variability risk.",
        "interacts": "Strongly interacts with agitation.",
    },
    "Stoichiometric Ratio": {
        "what": "Represents the balance between key reactants or components.",
        "why": "Ratio control often defines whether the chemistry stays in the target zone.",
        "low": "Can leave the system underfed or imbalanced.",
        "high": "Can push off-target composition or create inefficiency.",
        "interacts": "Strongly interacts with pH and overall chemistry.",
    },
}


# -----------------------------
# Header
# -----------------------------
st.title("🧪 Batch Behavior Explorer")
st.caption(
    "Interactive learning tool for batch process variables. "
    "This version uses educational heuristics to help you explore cause-and-effect, not plant-validated predictions."
)

with st.expander("How to think about this tool"):
    st.markdown(
        """
- Use it to learn **which variables matter most**.
- Explore **tradeoffs** between throughput, consistency, and risk.
- Compare two operating ideas side by side.
- Use the sensitivity view to see which knob changes the outcome fastest.

This is ideal for building intuition before you have a full data-driven model.
"""
    )

default_scenario = {
    "temp_c": 82.0,
    "ph": 5.80,
    "agitation_rpm": 240,
    "feed_rate": 1.20,
    "hold_time_min": 95,
    "solids_pct": 32.0,
    "ratio": 1.05,
}

tabs = st.tabs(
    [
        "Learn the Variables",
        "Scenario Simulator",
        "Compare Two Runs",
        "Sensitivity Lab",
    ]
)


# -----------------------------
# Tab 1 - Learn the Variables
# -----------------------------
with tabs[0]:
    st.subheader("Learn the Variables")

    var = st.selectbox("Choose a variable", list(VARIABLE_GUIDE.keys()))
    info = VARIABLE_GUIDE[var]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {var}")
        st.markdown(f"**What it is:** {info['what']}")
        st.markdown(f"**Why it matters:** {info['why']}")
        st.markdown(f"**Main interaction:** {info['interacts']}")
    with c2:
        st.markdown("### Operating intuition")
        st.markdown(f"**Too low:** {info['low']}")
        st.markdown(f"**Too high:** {info['high']}")

    st.markdown("### Typical preferred region in this learning model")
    preferred_df = pd.DataFrame(
        {
            "Variable": [
                "Temperature",
                "pH",
                "Agitation",
                "Feed Rate",
                "Hold Time",
                "Solids",
                "Stoichiometric Ratio",
            ],
            "Preferred Region": [
                "78–86 °C",
                "5.5–6.1",
                "200–290 rpm",
                "1.0–1.4",
                "80–110 min",
                "26–36%",
                "0.98–1.12",
            ],
        }
    )
    st.dataframe(preferred_df, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 2 - Scenario Simulator
# -----------------------------
with tabs[1]:
    st.subheader("Scenario Simulator")

    left, right = st.columns([1, 1.5])

    with left:
        st.markdown("### Inputs")
        scenario = scenario_inputs("sim", default_scenario)
        result = evaluate_scenario(scenario)

    with right:
        st.markdown("### Headline KPIs")
        metric_card_row(result)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_radar(result), use_container_width=True)
        with c2:
            st.plotly_chart(make_profile_plot(scenario), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.plotly_chart(make_safe_zone_map(scenario), use_container_width=True)

    with c4:
        st.markdown("### What this scenario is telling you")
        st.markdown(f"**Overall Health Score:** {result['overall_score']}/100")

        st.markdown("**Most sensitive weak spots right now:**")
        for name, score in result["weakest_drivers"]:
            st.write(f"- {name}: {round(score, 1)}/100")

        st.markdown("**Interpretation notes:**")
        for note in result["notes"]:
            st.write(f"- {note}")

        if result["penalties"]:
            st.markdown("**Tradeoffs / penalties detected:**")
            for item in result["penalties"]:
                st.write(f"- {item[0]}")
        else:
            st.success("No major interaction penalties triggered in this scenario.")


# -----------------------------
# Tab 3 - Compare Two Runs
# -----------------------------
with tabs[2]:
    st.subheader("Compare Two Runs")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Run A")
        scenario_a = scenario_inputs("run_a", default_scenario)
        result_a = evaluate_scenario(scenario_a)

    with col_b:
        st.markdown("### Run B")
        defaults_b = {
            "temp_c": 88.0,
            "ph": 5.55,
            "agitation_rpm": 210,
            "feed_rate": 1.45,
            "hold_time_min": 110,
            "solids_pct": 37.0,
            "ratio": 1.12,
        }
        scenario_b = scenario_inputs("run_b", defaults_b)
        result_b = evaluate_scenario(scenario_b)

    st.markdown("### KPI Comparison")
    compare_df = compare_dataframe(result_a, result_b)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    chart_df = compare_df.melt(
        id_vars="Metric",
        value_vars=["Run A", "Run B"],
        var_name="Run",
        value_name="Value",
    )
    fig_compare = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        color="Run",
        barmode="group",
        title="Run A vs Run B",
    )
    fig_compare.update_layout(height=450)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("### Biggest differences")
    diffs = {
        "Temperature Δ": round(scenario_b.temp_c - scenario_a.temp_c, 2),
        "pH Δ": round(scenario_b.ph - scenario_a.ph, 2),
        "Agitation Δ": round(scenario_b.agitation_rpm - scenario_a.agitation_rpm, 2),
        "Feed Rate Δ": round(scenario_b.feed_rate - scenario_a.feed_rate, 2),
        "Hold Time Δ": round(scenario_b.hold_time_min - scenario_a.hold_time_min, 2),
        "Solids Δ": round(scenario_b.solids_pct - scenario_a.solids_pct, 2),
        "Ratio Δ": round(scenario_b.ratio - scenario_a.ratio, 2),
    }
    diff_df = pd.DataFrame({"Change": list(diffs.keys()), "Delta": list(diffs.values())})
    st.dataframe(diff_df, use_container_width=True, hide_index=True)


# -----------------------------
# Tab 4 - Sensitivity Lab
# -----------------------------
with tabs[3]:
    st.subheader("Sensitivity Lab")

    st.markdown(
        "Pick one variable to sweep while the rest stay fixed. "
        "This helps show which knobs move yield, consistency, and risk the most."
    )

    base_left, base_right = st.columns([1, 1.4])

    with base_left:
        st.markdown("### Baseline Scenario")
        base = scenario_inputs("sens_base", default_scenario)

        variable = st.selectbox(
            "Variable to sweep",
            [
                "Temperature",
                "pH",
                "Agitation",
                "Feed Rate",
                "Hold Time",
                "Solids",
                "Stoichiometric Ratio",
            ],
        )

    sweep_map = {
        "Temperature": ("temp_c", np.linspace(60, 105, 40)),
        "pH": ("ph", np.linspace(4.5, 7.0, 40)),
        "Agitation": ("agitation_rpm", np.linspace(80, 400, 40)),
        "Feed Rate": ("feed_rate", np.linspace(0.60, 2.00, 40)),
        "Hold Time": ("hold_time_min", np.linspace(30, 180, 40)),
        "Solids": ("solids_pct", np.linspace(10, 50, 40)),
        "Stoichiometric Ratio": ("ratio", np.linspace(0.70, 1.40, 40)),
    }

    attr_name, sweep_values = sweep_map[variable]

    records = []
    for val in sweep_values:
        test = Scenario(
            temp_c=base.temp_c,
            ph=base.ph,
            agitation_rpm=base.agitation_rpm,
            feed_rate=base.feed_rate,
            hold_time_min=base.hold_time_min,
            solids_pct=base.solids_pct,
            ratio=base.ratio,
        )
        setattr(test, attr_name, float(val))
        result = evaluate_scenario(test)
        records.append(
            {
                "Variable Value": float(val),
                "Yield Score": result["yield_score"],
                "Consistency Score": result["consistency_score"],
                "Risk Score": result["risk_score"],
                "Overall Score": result["overall_score"],
            }
        )

    sens_df = pd.DataFrame(records)

    with base_right:
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=sens_df["Variable Value"], y=sens_df["Yield Score"], mode="lines", name="Yield"))
        fig_sens.add_trace(
            go.Scatter(x=sens_df["Variable Value"], y=sens_df["Consistency Score"], mode="lines", name="Consistency")
        )
        fig_sens.add_trace(go.Scatter(x=sens_df["Variable Value"], y=sens_df["Risk Score"], mode="lines", name="Risk"))
        fig_sens.add_trace(
            go.Scatter(x=sens_df["Variable Value"], y=sens_df["Overall Score"], mode="lines", name="Overall")
        )

        current_value = getattr(base, attr_name)
        fig_sens.add_vline(x=current_value, line_dash="dash")
        fig_sens.update_layout(
            title=f"Sensitivity Sweep: {variable}",
            xaxis_title=variable,
            yaxis_title="Score",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    best_idx = sens_df["Overall Score"].idxmax()
    best_row = sens_df.loc[best_idx]

    st.markdown("### Best point found in this sweep")
    st.write(
        f"- **Variable:** {variable}\n"
        f"- **Best Value:** {round(best_row['Variable Value'], 3)}\n"
        f"- **Overall Score:** {round(best_row['Overall Score'], 1)}"
    )

    st.markdown("### Sweep data")
    st.dataframe(sens_df.round(2), use_container_width=True, hide_index=True)