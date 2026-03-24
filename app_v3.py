
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Zinpro Pilot Plant Dashboard v3", layout="wide")

def mad(x):
    s = pd.Series(x).dropna()
    if s.empty:
        return np.nan
    med = np.median(s)
    return float(np.median(np.abs(s - med)))

def pct(series, q):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(np.nanpercentile(s, q))

def fmt(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"

def latest_nonempty(series):
    s = pd.Series(series).fillna("").astype(str)
    s = s[s.str.strip() != ""]
    return "" if s.empty else s.iloc[-1]

def in_spec_rate(series, low=None, high=None):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    mask = pd.Series(True, index=s.index)
    if low is not None:
        mask &= s >= low
    if high is not None:
        mask &= s <= high
    return float(mask.mean() * 100)

def compute_summary(df, pH_low, pH_high, temp_low, temp_high, pH_std_limit, temp_std_limit, alarm_limit):
    g = df.groupby("batch_id", dropna=False)

    summary = g.agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        duration_min=("timestamp", lambda s: (s.max() - s.min()).total_seconds()/60 if len(s) else np.nan),
        pH_avg=("pH", "mean"),
        pH_std=("pH", "std"),
        pH_median=("pH", "median"),
        pH_mad=("pH", mad),
        temp_avg=("temperature_C", "mean"),
        temp_std=("temperature_C", "std"),
        temp_median=("temperature_C", "median"),
        metal_feed_avg=("metal_feed_rate", "mean"),
        aa_feed_avg=("aa_feed_rate", "mean"),
        mixer_rpm_avg=("mixer_rpm", "mean"),
        last_event=("event", latest_nonempty),
        batch_release_decision=("batch_release_decision", latest_nonempty),
        operator_notes=("operator_notes", latest_nonempty),
        post_maint_verification=("post_maintenance_verification", latest_nonempty),
        calibration_status=("calibration_status", latest_nonempty),
        handoff_to_rnd=("handoff_to_rnd", latest_nonempty),
        handoff_to_ops=("handoff_to_ops", latest_nonempty),
        handoff_to_leadership=("handoff_to_leadership", latest_nonempty),
    ).reset_index()

    if "product_yield_kg" in df.columns:
        y = g["product_yield_kg"].max().reset_index(name="product_yield_kg")
        summary = summary.merge(y, on="batch_id", how="left")

    if "alarm" in df.columns:
        alarms = (
            df.assign(has_alarm=df["alarm"].fillna("").astype(str).str.strip() != "")
            .groupby("batch_id")["has_alarm"].sum()
            .reset_index(name="alarm_count")
        )
        summary = summary.merge(alarms, on="batch_id", how="left")
    else:
        summary["alarm_count"] = 0

    pH_spec = df.groupby("batch_id")["pH"].apply(lambda s: in_spec_rate(s, pH_low, pH_high)).reset_index(name="pH_in_spec_pct")
    temp_spec = df.groupby("batch_id")["temperature_C"].apply(lambda s: in_spec_rate(s, temp_low, temp_high)).reset_index(name="temp_in_spec_pct")
    summary = summary.merge(pH_spec, on="batch_id", how="left").merge(temp_spec, on="batch_id", how="left")

    summary["control_status"] = np.where(
        (summary["pH_std"].fillna(999) <= pH_std_limit) &
        (summary["temp_std"].fillna(999) <= temp_std_limit) &
        (summary["alarm_count"].fillna(999) <= alarm_limit),
        "Stable",
        "Needs Review"
    )

    summary["data_validity_status"] = np.where(
        summary["calibration_status"].fillna("").str.upper().isin(["VERIFIED", "OK", "CURRENT"]),
        "Trusted",
        "Check Instruments"
    )

    summary["overall_status"] = np.where(
        (summary["control_status"] == "Stable") &
        (summary["data_validity_status"] == "Trusted"),
        "Ready for Review",
        "Hold / Review"
    )

    return summary

def make_release_text(row):
    return "\n".join([
        f"Batch: {row['batch_id']}",
        f"Time window: {row['start_time']} to {row['end_time']}",
        f"Overall status: {row['overall_status']}",
        f"Control status: {row['control_status']}",
        f"Data validity: {row['data_validity_status']}",
        f"Batch release decision: {row['batch_release_decision'] or 'Pending'}",
        f"Post-maintenance verification: {row['post_maint_verification'] or 'Not logged'}",
        f"pH median / std: {fmt(row['pH_median'],3)} / {fmt(row['pH_std'],3)}",
        f"Temp avg / std: {fmt(row['temp_avg'],2)} °C / {fmt(row['temp_std'],2)} °C",
        f"Alarm count: {int(row['alarm_count']) if not pd.isna(row['alarm_count']) else 0}",
        f"Operator note: {row['operator_notes'] or 'None'}",
        f"R&D handoff: {row['handoff_to_rnd'] or 'None'}",
        f"Operations handoff: {row['handoff_to_ops'] or 'None'}",
        f"Leadership handoff: {row['handoff_to_leadership'] or 'None'}",
    ])

st.title("Zinpro Pilot Plant Dashboard v3")
st.caption("Tailored for batch release, post-maintenance verification, operator notes, and pilot-to-production handoff.")

with st.sidebar:
    st.header("Control Limits")
    pH_low = st.number_input("Target pH low", value=5.45, step=0.01, format="%.2f")
    pH_high = st.number_input("Target pH high", value=5.70, step=0.01, format="%.2f")
    temp_low = st.number_input("Target temp low (°C)", value=60.0, step=0.5, format="%.1f")
    temp_high = st.number_input("Target temp high (°C)", value=64.0, step=0.5, format="%.1f")
    pH_std_limit = st.number_input("pH std dev limit", value=0.08, step=0.01, format="%.2f")
    temp_std_limit = st.number_input("Temp std dev limit (°C)", value=1.50, step=0.10, format="%.2f")
    alarm_limit = st.number_input("Alarm count limit", value=2, step=1)

uploaded = st.file_uploader("Upload batch CSV", type=["csv"])
st.markdown("Expected columns include standard process fields plus: `batch_release_decision, post_maintenance_verification, operator_notes, handoff_to_rnd, handoff_to_ops, handoff_to_leadership`")

if uploaded is None:
    st.info("Upload the sample CSV included in the package to try the dashboard.")
    st.stop()

df = pd.read_csv(uploaded)
required = ["timestamp","batch_id","pH","temperature_C","metal_feed_rate","aa_feed_rate","mixer_rpm"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["batch_id","timestamp"]).copy()

summary = compute_summary(df, pH_low, pH_high, temp_low, temp_high, pH_std_limit, temp_std_limit, alarm_limit)

selected_batch = st.selectbox("Select batch", summary["batch_id"].tolist(), index=len(summary)-1)
live = df[df["batch_id"] == selected_batch].copy()
row = summary[summary["batch_id"] == selected_batch].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Batch", str(selected_batch))
c2.metric("Overall status", row["overall_status"])
c3.metric("Control status", row["control_status"])
c4.metric("Data validity", row["data_validity_status"])
c5.metric("Batch release", row["batch_release_decision"] or "Pending")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Process",
    "Batch Release",
    "Post-Maintenance Verification",
    "Operator Notes",
    "Pilot-to-Production Handoff",
    "Trend View"
])

with tab1:
    a, b = st.columns(2)
    with a:
        st.subheader("pH trend")
        st.line_chart(live.set_index("timestamp")[["pH"]])
        st.caption(f"Target band: {pH_low:.2f} to {pH_high:.2f}")
    with b:
        st.subheader("Temperature trend")
        st.line_chart(live.set_index("timestamp")[["temperature_C"]])
        st.caption(f"Target band: {temp_low:.1f} to {temp_high:.1f} °C")

    c, d = st.columns(2)
    with c:
        st.subheader("Feed rates")
        st.line_chart(live.set_index("timestamp")[["metal_feed_rate","aa_feed_rate"]])
    with d:
        st.subheader("Mixer RPM")
        st.line_chart(live.set_index("timestamp")[["mixer_rpm"]])

    st.subheader("Recent alarms")
    alarm_rows = live[live.get("alarm", pd.Series(index=live.index, dtype=str)).fillna("").astype(str).str.strip() != ""]
    if len(alarm_rows):
        st.dataframe(alarm_rows[["timestamp","alarm","event"]].tail(10), use_container_width=True)
    else:
        st.success("No alarms recorded for this batch.")

with tab2:
    a, b = st.columns([1.2,1])
    with a:
        st.subheader("Batch release scorecard")
        scorecard = pd.DataFrame({
            "KPI": [
                "Overall status","Control status","Data validity","Release decision",
                "pH in spec %","Temp in spec %","pH std dev","Temp std dev","Alarm count","Yield (kg)"
            ],
            "Value": [
                row["overall_status"], row["control_status"], row["data_validity_status"], row["batch_release_decision"] or "Pending",
                fmt(row["pH_in_spec_pct"],1), fmt(row["temp_in_spec_pct"],1), fmt(row["pH_std"],3), fmt(row["temp_std"],2),
                int(row["alarm_count"]) if not pd.isna(row["alarm_count"]) else 0,
                fmt(row["product_yield_kg"],2) if "product_yield_kg" in row.index else "—"
            ]
        })
        st.dataframe(scorecard, use_container_width=True)
    with b:
        st.subheader("Release logic")
        st.markdown(f"""
- **Control:** {row['control_status']}
- **Data validity:** {row['data_validity_status']}
- **Decision:** {row['batch_release_decision'] or 'Pending'}
- **Why this matters:** only stable, trusted batches should drive pilot conclusions or scale-up decisions.
""")

with tab3:
    st.subheader("Post-maintenance verification")
    st.markdown(f"**Verification status:** {row['post_maint_verification'] or 'Not logged'}")
    st.markdown("""
Use this section to show your mindset:
- confirm instruments are stable after maintenance
- verify no drift before normal operation
- do not quietly return the system to service without a check
""")
    if "event" in live.columns:
        maint_like = live[live["event"].fillna("").astype(str).str.contains("maint", case=False, na=False)]
        if len(maint_like):
            st.dataframe(maint_like[["timestamp","event"]], use_container_width=True)

with tab4:
    a, b = st.columns(2)
    with a:
        st.subheader("Operator note")
        st.text_area("Operator note", value=row["operator_notes"] or "", height=180)
    with b:
        st.subheader("Latest event")
        st.markdown(f"**{row['last_event'] or 'No event logged'}**")
        st.markdown("""
Use notes to capture:
- unusual mixing behavior
- feed interruptions
- instrument concerns
- visual observations
- anything that could explain why a batch drifted
""")

with tab5:
    st.subheader("Cross-functional handoff")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**R&D**")
        st.text_area("R&D handoff", value=row["handoff_to_rnd"] or "", height=180, key="rnd")
    with b:
        st.markdown("**Operations**")
        st.text_area("Operations handoff", value=row["handoff_to_ops"] or "", height=180, key="ops")
    with c:
        st.markdown("**Leadership**")
        st.text_area("Leadership handoff", value=row["handoff_to_leadership"] or "", height=180, key="lead")

    st.subheader("Copy/paste summary")
    st.text_area("Release summary", value=make_release_text(row), height=260)

with tab6:
    trend = summary.sort_values("end_time").set_index("end_time")
    a, b = st.columns(2)
    with a:
        st.subheader("pH stability by batch")
        st.line_chart(trend[["pH_std"]])
    with b:
        st.subheader("Temperature stability by batch")
        st.line_chart(trend[["temp_std"]])

    c, d = st.columns(2)
    with c:
        st.subheader("Spec compliance")
        comp = summary[["batch_id","pH_in_spec_pct","temp_in_spec_pct"]].set_index("batch_id")
        st.bar_chart(comp)
    with d:
        if "product_yield_kg" in summary.columns:
            st.subheader("Yield vs pH median")
            plot_df = summary[["pH_median","product_yield_kg"]].dropna()
            st.scatter_chart(plot_df, x="pH_median", y="product_yield_kg")

st.divider()
st.markdown("""
### Interview framing
- I thought about the pilot plant as a consistency and handoff system.
- Even if the reactor hardware is already designed, the way batches are run, verified, released, and communicated still creates a lot of value.
- My role would be to make sure the data is trustworthy and the process is repeatable before anyone makes scale-up decisions.
""")
