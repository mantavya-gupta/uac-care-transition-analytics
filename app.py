"""
Care Transition Efficiency & Placement Outcome Analytics
Streamlit dashboard for the HHS Unaccompanied Alien Children (UAC) Program dataset.

Run with:  streamlit run app.py
Expects uac_data.csv (cleaned) in the same directory.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config & style
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="UAC Care Transition Analytics",
    page_icon="\U0001F9ED",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1F3864"
TEAL = "#1F6F8B"
ACCENT = "#E8743B"
GREEN = "#3A9D5D"
GREY = "#7A7A7A"
LIGHTBG = "#EAF1F4"

st.markdown(
    f"""
    <style>
    .main {{ background-color: #FAFBFC; }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E3E8EC;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .alert-ok {{
        background-color: #E9F7EF; border-left: 5px solid {GREEN};
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; color:#1e4620; font-size:0.92rem;
    }}
    .alert-warn {{
        background-color: #FDF3E7; border-left: 5px solid {ACCENT};
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; color:#6b3a10; font-size:0.92rem;
    }}
    .section-note {{
        color: {GREY}; font-size: 0.88rem; margin-top: -6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "uac_data.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["transfer_efficiency_ratio"] = (df["transferred_out"] / df["cbp_custody"]).replace([np.inf, -np.inf], np.nan)
    df["discharge_effectiveness"] = (df["discharged"] / df["hhs_care"]).replace([np.inf, -np.inf], np.nan)
    df["net_flow"] = df["transferred_out"] - df["discharged"]
    df["backlog_accum_rate"] = (df["net_flow"] / df["hhs_care"]).replace([np.inf, -np.inf], np.nan)

    df["roll_apprehended_30"] = df["apprehended"].rolling(30, min_periods=15).sum()
    df["roll_discharged_30"] = df["discharged"].rolling(30, min_periods=15).sum()
    df["pipeline_throughput_30"] = (df["roll_discharged_30"] / df["roll_apprehended_30"]).replace([np.inf, -np.inf], np.nan)

    roll_de = df["discharge_effectiveness"].rolling(30, min_periods=15)
    df["outcome_stability_30"] = 1 - (roll_de.std() / roll_de.mean())

    df["transfer_eff_7d"] = df["transfer_efficiency_ratio"].rolling(7, min_periods=3).mean()
    df["discharge_eff_7d"] = df["discharge_effectiveness"].rolling(7, min_periods=3).mean()
    df["backlog_rate_7d"] = df["backlog_accum_rate"].rolling(7, min_periods=3).mean()

    df["weekday"] = df["date"].dt.day_name()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

df_all = load_data()

# --------------------------------------------------------------------------------------
# Sidebar — controls
# --------------------------------------------------------------------------------------
st.sidebar.title("\U0001F9ED  Controls")

min_d, max_d = df_all["date"].min().date(), df_all["date"].max().date()
st.sidebar.subheader("Date range")
date_range = st.sidebar.date_input(
    "Reporting period",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

st.sidebar.subheader("Rolling window for ratios")
window = st.sidebar.radio("Smoothing", ["Daily (raw)", "7-day average", "30-day average"], index=1)

st.sidebar.subheader("Metrics to display")
show_transfer = st.sidebar.checkbox("Transfer Efficiency Ratio", value=True)
show_discharge = st.sidebar.checkbox("Discharge Effectiveness Index", value=True)
show_throughput = st.sidebar.checkbox("Pipeline Throughput Rate", value=True)
show_backlog = st.sidebar.checkbox("Backlog Accumulation Rate", value=True)
show_stability = st.sidebar.checkbox("Outcome Stability Score", value=True)

st.sidebar.subheader("Alert thresholds")
te_threshold = st.sidebar.slider("Transfer Efficiency Ratio — warn below", 0.0, 1.5, 0.5, 0.05)
de_threshold = st.sidebar.slider("Discharge Effectiveness — warn below (%)", 0.0, 10.0, 1.5, 0.1) / 100
backlog_streak_threshold = st.sidebar.slider("Backlog streak alert (consecutive days)", 3, 20, 7, 1)
stability_threshold = st.sidebar.slider("Outcome Stability Score — warn below", 0.0, 1.0, 0.6, 0.05)

mask = (df_all["date"].dt.date >= start_d) & (df_all["date"].dt.date <= end_d)
df = df_all.loc[mask].copy()

if df.empty:
    st.warning("No reporting records fall inside the selected date range. Widen the range in the sidebar.")
    st.stop()

def series_for(col_daily, col_7d):
    if window == "Daily (raw)":
        return df[col_daily]
    elif window == "7-day average":
        return df[col_7d] if col_7d in df.columns else df[col_daily].rolling(7, min_periods=3).mean()
    else:
        return df[col_daily].rolling(30, min_periods=15).mean()

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.markdown(
    f"<span class='section-note'>HHS Unaccompanied Alien Children (UAC) Program &nbsp;|&nbsp; "
    f"Showing {len(df):,} reporting records from {start_d} to {end_d}</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --------------------------------------------------------------------------------------
# KPI scorecards
# --------------------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
avg_te = df["transfer_efficiency_ratio"].mean()
avg_de = df["discharge_effectiveness"].mean()
avg_pt = df["pipeline_throughput_30"].mean()
avg_bar = df["backlog_accum_rate"].mean()
avg_oss = df["outcome_stability_30"].mean()

c1.metric("Transfer Efficiency Ratio", f"{avg_te:.2f}" if pd.notna(avg_te) else "—")
c2.metric("Discharge Effectiveness", f"{avg_de*100:.1f}%" if pd.notna(avg_de) else "—")
c3.metric("Pipeline Throughput (30d)", f"{avg_pt:.2f}" if pd.notna(avg_pt) else "—")
c4.metric("Backlog Accum. Rate", f"{avg_bar*100:+.2f}%/day" if pd.notna(avg_bar) else "—")
c5.metric("Outcome Stability Score", f"{avg_oss:.2f}" if pd.notna(avg_oss) else "—")

st.markdown("---")

# --------------------------------------------------------------------------------------
# Threshold alerts
# --------------------------------------------------------------------------------------
st.subheader("\U0001F6A8 Threshold Alerts")

kpi_alerts = []
if pd.notna(avg_te) and avg_te < te_threshold:
    kpi_alerts.append(f"Transfer Efficiency Ratio averages {avg_te:.2f} in this period, below your {te_threshold:.2f} threshold — CBP custody may be clearing slower than target.")
if pd.notna(avg_de) and avg_de < de_threshold:
    kpi_alerts.append(f"Discharge Effectiveness averages {avg_de*100:.1f}% in this period, below your {de_threshold*100:.1f}% threshold — sponsor placements are clearing a smaller share of the HHS census than target.")
if pd.notna(avg_oss) and avg_oss < stability_threshold:
    kpi_alerts.append(f"Outcome Stability Score averages {avg_oss:.2f}, below your {stability_threshold:.2f} threshold — placement outcomes are less consistent than target in this period.")

# backlog streak detection within selected range (reused by the Bottleneck Detection tab below)
# NOTE: the source feed is not gap-free (see README/Research Paper Sec. 5.1 — only 720 of 1,075
# calendar days in range have a record, and Fridays/Saturdays are almost entirely missing). "length"
# below counts consecutive *reporting* records, which can span a longer *calendar* range whenever a
# gap falls inside the streak. Both figures are shown in the tables below to avoid that reading as an error.
df["backlog_flag"] = df["net_flow"] > 0
streak_id = (df["backlog_flag"] != df["backlog_flag"].shift()).cumsum()
streaks = df.groupby(streak_id).agg(flag=("backlog_flag", "first"), length=("backlog_flag", "size"),
                                     start=("date", "first"), end=("date", "last"))
streaks["calendar_days"] = (streaks["end"] - streaks["start"]).dt.days + 1
long_streaks = streaks[(streaks.flag) & (streaks.length >= backlog_streak_threshold)].sort_values("length", ascending=False)

any_alert = bool(kpi_alerts) or len(long_streaks) > 0

for msg in kpi_alerts:
    st.markdown(f"<div class='alert-warn'>\u26A0\uFE0F {msg}</div>", unsafe_allow_html=True)

if len(long_streaks) > 0:
    worst = long_streaks.iloc[0]
    summary = (f"{len(long_streaks)} backlog build-up streak(s) of \u2265{backlog_streak_threshold} consecutive reporting days found in this period "
               f"(longest: {int(worst['length'])} reporting days, {worst['start'].date()} to {worst['end'].date()}).")
    st.markdown(f"<div class='alert-warn'>\u26A0\uFE0F {summary}</div>", unsafe_allow_html=True)
    with st.expander(f"See all {len(long_streaks)} backlog streak(s)"):
        st.dataframe(
            long_streaks[["start", "end", "length", "calendar_days"]].rename(
                columns={"start": "Start", "end": "End", "length": "Consecutive reporting days", "calendar_days": "Calendar days spanned"}
            ),
            use_container_width=True, hide_index=True,
        )
        if (long_streaks["calendar_days"] != long_streaks["length"]).any():
            st.caption("\u2139\uFE0F \u201cConsecutive reporting days\u201d and \u201cCalendar days spanned\u201d differ when a gap in the source feed "
                       "(mostly missing Fridays/Saturdays \u2014 see README) falls inside the streak. This isn't a calculation error.")

if not any_alert:
    st.markdown("<div class='alert-ok'>\u2705 No thresholds breached in the selected period and settings.</div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------------------------------------------
# Tabs — core modules
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F9ED Care Pipeline Flow",
    "\u2696\uFE0F Transfer & Discharge Efficiency",
    "\U0001F6A7 Bottleneck Detection",
    "\U0001F4C8 Outcome Trend Analysis",
])

# ---- Tab 1: Care Pipeline Flow Visualization ----
with tab1:
    st.subheader("Care Pipeline Flow")
    st.caption("Daily volumes moving through the CBP custody \u2192 HHS care \u2192 sponsor placement pipeline.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["apprehended"], name="Apprehended (into CBP)", line=dict(color=TEAL, width=1.6)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["transferred_out"], name="Transferred (CBP\u2192HHS)", line=dict(color=ACCENT, width=1.6)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["discharged"], name="Discharged (HHS\u2192Sponsor)", line=dict(color=GREEN, width=1.6)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                       plot_bgcolor="white", yaxis_title="Children / day")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pipeline flow diagram** — average children/day moving between stages in the selected period")
    avg_appr = df["apprehended"].mean()
    avg_trans = df["transferred_out"].mean()
    avg_disch = df["discharged"].mean()
    sankey = go.Figure(go.Sankey(
        node=dict(
            label=[
                f"Apprehended ({avg_appr:,.0f}/day)",
                "CBP Custody",
                "HHS Care",
                f"Discharged to Sponsor ({avg_disch:,.0f}/day)",
            ],
            color=[TEAL, TEAL, NAVY, GREEN],
            pad=25, thickness=18,
        ),
        link=dict(
            source=[0, 1, 2],
            target=[1, 2, 3],
            value=[max(avg_appr, 0.01), max(avg_trans, 0.01), max(avg_disch, 0.01)],
            color=["rgba(31,111,139,0.35)", "rgba(232,116,59,0.45)", "rgba(58,157,93,0.4)"],
        ),
    ))
    sankey.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=13))
    st.plotly_chart(sankey, use_container_width=True)
    st.caption("Node width reflects average daily volume in the selected period. This shows stage-to-stage throughput rates, not linked individual-child journeys.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Active stock levels**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["cbp_custody"], name="In CBP custody", fill="tozeroy",
                                   line=dict(color=TEAL, width=1)))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                            yaxis_title="Children in CBP custody")
        fig2.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        st.markdown("**HHS care census**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["date"], y=df["hhs_care"], name="In HHS care", fill="tozeroy",
                                   line=dict(color=NAVY, width=1)))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                            yaxis_title="Children in HHS care")
        fig3.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Stage snapshot (period average)**")
    stage_avg = pd.DataFrame({
        "Stage": ["Apprehended /day", "In CBP custody", "Transferred out /day", "In HHS care", "Discharged /day"],
        "Average": [df["apprehended"].mean(), df["cbp_custody"].mean(), df["transferred_out"].mean(),
                    df["hhs_care"].mean(), df["discharged"].mean()],
    })
    figbar = go.Figure(go.Bar(x=stage_avg["Stage"], y=stage_avg["Average"],
                               marker_color=[TEAL, TEAL, ACCENT, NAVY, GREEN]))
    figbar.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="Children")
    figbar.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(figbar, use_container_width=True)

# ---- Tab 2: Transfer & Discharge Efficiency Panels ----
with tab2:
    st.subheader("Transfer & Discharge Efficiency")
    st.caption(f"Smoothing: {window}. Toggle metrics in the sidebar.")

    if show_transfer:
        st.markdown("**Transfer Efficiency Ratio** — Transferred out \u00f7 CBP custody")
        y = series_for("transfer_efficiency_ratio", "transfer_eff_7d")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=y, name="Transfer Efficiency Ratio", line=dict(color=TEAL, width=1.8)))
        fig.add_hline(y=te_threshold, line_dash="dash", line_color=ACCENT,
                      annotation_text=f"Alert threshold ({te_threshold:.2f})", annotation_position="top left")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        fig.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig, use_container_width=True)

    if show_discharge:
        st.markdown("**Discharge Effectiveness Index** — Discharged \u00f7 HHS care")
        y = series_for("discharge_effectiveness", "discharge_eff_7d") * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=y, name="Discharge Effectiveness (%)", line=dict(color=ACCENT, width=1.8)))
        fig.add_hline(y=de_threshold * 100, line_dash="dash", line_color=TEAL,
                      annotation_text=f"Alert threshold ({de_threshold*100:.1f}%)", annotation_position="top left")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="%")
        fig.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig, use_container_width=True)

    if show_throughput:
        st.markdown("**Pipeline Throughput Rate** — 30-day rolling discharges \u00f7 30-day rolling apprehensions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["pipeline_throughput_30"], name="Pipeline Throughput (30d)",
                                  line=dict(color=GREEN, width=1.8)))
        fig.add_hline(y=1.0, line_dash="dot", line_color=GREY, annotation_text="Parity (exits = entries)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        fig.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Daily (unsmoothed) throughput is highly volatile because discharges reflect children who often entered weeks earlier; the 30-day rolling version is used here for a meaningful read.")

    st.markdown("**Weekday pattern (this selection)**")
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = df.groupby("weekday")[["transfer_efficiency_ratio", "discharge_effectiveness"]].mean().reindex(wd_order)
    counts = df["weekday"].value_counts().reindex(wd_order).fillna(0).astype(int)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=wd_order, y=wd["transfer_efficiency_ratio"], name="Transfer Efficiency Ratio", marker_color=TEAL), secondary_y=False)
    fig.add_trace(go.Bar(x=wd_order, y=wd["discharge_effectiveness"] * 100, name="Discharge Effectiveness (%)", marker_color=ACCENT), secondary_y=True)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), barmode="group", plot_bgcolor="white",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    low_days = counts[counts < counts.max() * 0.1]
    if len(low_days) > 0:
        st.caption(f"Note: {', '.join(low_days.index)} have very few reporting records ({', '.join(str(v) for v in low_days.values)} respectively) in the selected range — treat those weekday averages with caution.")

# ---- Tab 3: Bottleneck Detection ----
with tab3:
    st.subheader("Bottleneck Detection")
    st.caption("Net flow into HHS care (transfers in minus discharges out). Positive bars mean the same-day HHS census is growing.")

    colors = np.where(df["net_flow"] >= 0, ACCENT, TEAL)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["net_flow"], marker_color=colors, name="Net flow"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="Net children / day")
    fig.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    if show_backlog:
        st.markdown("**Backlog Accumulation Rate** — Net flow \u00f7 current HHS census")
        y = series_for("backlog_accum_rate", "backlog_rate_7d") * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=y, line=dict(color=NAVY, width=1.6), name="Backlog Accum. Rate (%)"))
        fig.add_hline(y=0, line_dash="dot", line_color=GREY)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_title="% of census / day")
        fig.update_yaxes(gridcolor="#EEEEEE")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Sustained backlog build-up streaks (\u2265 {backlog_streak_threshold} consecutive reporting days)**")
    if len(long_streaks) > 0:
        show_tbl = long_streaks[["start", "end", "length", "calendar_days"]].rename(
            columns={"start": "Start", "end": "End", "length": "Consecutive reporting days", "calendar_days": "Calendar days spanned"}
        ).sort_values("Consecutive reporting days", ascending=False)
        st.dataframe(show_tbl, use_container_width=True, hide_index=True)
        st.caption("Reporting days vs. calendar days can differ because the source feed has gaps (mostly missing Fridays/Saturdays). See README for details.")
    else:
        st.info(f"No streak of {backlog_streak_threshold}+ consecutive backlog-building days in the selected period. Try lowering the threshold in the sidebar.")

# ---- Tab 4: Outcome Trend Analysis ----
with tab4:
    st.subheader("Outcome Trend Analysis")

    if show_stability:
        st.markdown("**Outcome Stability Score** — 1 \u2212 (30-day rolling std \u00f7 mean) of Discharge Effectiveness")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["outcome_stability_30"], line=dict(color=ACCENT, width=1.8), name="Outcome Stability Score"))
        fig.add_hline(y=stability_threshold, line_dash="dash", line_color=TEAL,
                      annotation_text=f"Alert threshold ({stability_threshold:.2f})", annotation_position="top left")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        fig.update_yaxes(gridcolor="#EEEEEE", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher = more consistent, predictable discharge performance. Lower = more erratic outcomes, even if the average level is unchanged.")

    st.markdown("**Monthly placement trend**")
    monthly = df.groupby("month").agg(discharged=("discharged", "mean"), hhs_care=("hhs_care", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["discharged"], name="Avg. discharged/day", marker_color=GREEN), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["hhs_care"], name="Avg. HHS census", line=dict(color=NAVY, width=2)), secondary_y=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                       legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Year-over-year KPI comparison**")
    yearly = df.groupby("year").agg(
        transfer_efficiency_ratio=("transfer_efficiency_ratio", "mean"),
        discharge_effectiveness=("discharge_effectiveness", "mean"),
        pipeline_throughput_30=("pipeline_throughput_30", "mean"),
        outcome_stability_30=("outcome_stability_30", "mean"),
    ).round(3)
    yearly.columns = ["Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput (30d)", "Outcome Stability Score"]
    st.dataframe(yearly, use_container_width=True)

st.markdown("---")
st.caption(
    "Care Transition Efficiency & Placement Outcome Analytics \u2014 built on publicly reported HHS UAC Program daily data. "
    "KPI formulas: Transfer Efficiency Ratio = Transferred \u00f7 CBP custody; Discharge Effectiveness = Discharged \u00f7 HHS care; "
    "Pipeline Throughput = 30-day rolling discharges \u00f7 30-day rolling apprehensions; Backlog Accumulation Rate = Net flow \u00f7 HHS care; "
    "Outcome Stability Score = 1 \u2212 (30-day rolling coefficient of variation of Discharge Effectiveness)."
)
