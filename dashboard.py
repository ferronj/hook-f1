"""
Interactive Streamlit dashboard for F1 race simulation results.

Run: micromamba run -n f1-markov streamlit run dashboard.py

Auto-discovers all data/sim_*.json files and presents a race selector.
Supports any number of stage models + composite per race.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="F1 Race Simulation Dashboard",
    page_icon="🏎️",
    layout="wide",
)

# =====================================================================
# DATA LOADING — discover all sim files
# =====================================================================
@st.cache_data
def discover_races():
    """Scan data/ for sim_*.json files and return {race_name: data_dict}."""
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("sim_*.json"))
    races = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        races[d["race"]] = d
    return races


races = discover_races()

if not races:
    st.error("No simulation files found in `data/sim_*.json`.")
    st.stop()

# =====================================================================
# SIDEBAR — race selector
# =====================================================================
race_names = list(races.keys())
race_name = st.sidebar.selectbox(
    "🏁 Select Race",
    race_names,
    index=len(race_names) - 1,  # default to most recent
)

data = races[race_name]

# Build model labels dynamically from JSON
MODEL_KEYS = list(data["models"].keys())
MODEL_LABELS = {k: data["models"][k]["name"] for k in MODEL_KEYS}

# Determine which are "stage" models (not composite) for comparison charts
STAGE_KEYS = [k for k in MODEL_KEYS if k != "composite"]


def get_drivers_df(model_key):
    drivers = data["models"][model_key]["drivers"]
    df = pd.DataFrame(drivers)
    df["rank"] = range(1, len(df) + 1)
    return df


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(races)}** race(s) available")
st.sidebar.markdown(f"**{len(STAGE_KEYS)}** stage models + composite")
st.sidebar.markdown(f"Training: {data['training_years']}")

# =====================================================================
# HEADER
# =====================================================================
st.title(f"🏁 {data['race']} — Simulation Dashboard")
st.markdown(
    f"**{data['circuit']}** · {data['date']} · "
    f"**{data['n_sims']:,}** Monte Carlo simulations per model · "
    f"Training: {data['training_years']} · "
    f"Models: {len(STAGE_KEYS)} stages + composite"
)

# =====================================================================
# KEY STORYLINES — auto-generated from data
# =====================================================================
st.header("📰 Key Storylines")

# Use composite for storylines (most balanced view)
comp_key = "composite" if "composite" in MODEL_KEYS else MODEL_KEYS[-1]
ens = get_drivers_df(comp_key)

story_cols = st.columns(4)

# Story 1: Title fight — top 2 drivers by P(win)
with story_cols[0]:
    d1, d2 = ens.iloc[0], ens.iloc[1]
    st.markdown(f"**{d1['abbreviation']} vs {d2['abbreviation']}**")
    st.markdown(f"{d1['abbreviation']}: {d1['p_win']:.1%} win, {d1['p_podium']:.1%} podium")
    st.markdown(f"{d2['abbreviation']}: {d2['p_win']:.1%} win, {d2['p_podium']:.1%} podium")

# Story 2: Best of the rest — 3rd place driver
with story_cols[1]:
    d3 = ens.iloc[2]
    st.markdown(f"**Best of the Rest: {d3['name']}**")
    st.markdown(f"P(win): {d3['p_win']:.1%}")
    st.markdown(f"P(podium): {d3['p_podium']:.1%}")
    st.markdown(f"E[pos]: {d3['e_pos']:.1f}")

# Story 3: Constructor battle — top 2 constructors
with story_cols[2]:
    team_pod = {}
    for team in ens["team"].unique():
        td = ens[ens["team"] == team]
        team_pod[team] = 1 - np.prod(1 - td["p_podium"].values)
    top_teams = sorted(team_pod, key=team_pod.get, reverse=True)[:2]
    st.markdown(f"**{top_teams[0]} vs {top_teams[1]}**")
    st.markdown(f"{top_teams[0]}: {team_pod[top_teams[0]]:.1%} P(podium)")
    st.markdown(f"{top_teams[1]}: {team_pod[top_teams[1]]:.1%} P(podium)")

# Story 4: Biggest underdog — highest P(podium) outside top 5
with story_cols[3]:
    underdogs = ens.iloc[5:]  # outside top 5 by E[pos]
    if len(underdogs) > 0:
        best_ud = underdogs.sort_values("p_podium", ascending=False).iloc[0]
        st.markdown(f"**Underdog: {best_ud['name']}**")
        st.markdown(f"Ranked {best_ud['rank']}th overall")
        st.markdown(f"P(podium): {best_ud['p_podium']:.1%}")
        st.markdown(f"{best_ud['team']}")

# =====================================================================
# MODEL SELECTOR
# =====================================================================
st.divider()

# Default to composite
default_idx = MODEL_KEYS.index("composite") if "composite" in MODEL_KEYS else 0
model_key = st.radio(
    "Select Model",
    MODEL_KEYS,
    format_func=lambda x: MODEL_LABELS[x],
    horizontal=True,
    index=default_idx,
)

model_info = data["models"][model_key]
st.info(model_info["description"])
params = model_info["params"]
if params and isinstance(list(params.values())[0], (int, float)):
    param_str = " · ".join(f"**{k}** = {v:.3f}" for k, v in params.items())
else:
    param_str = " · ".join(f"**{k}** = {v}" for k, v in params.items())
st.markdown(f"Parameters: {param_str}")

df = get_drivers_df(model_key)

# =====================================================================
# PODIUM PREDICTION
# =====================================================================
st.header("🏆 Podium Prediction")
podium_cols = st.columns(3)
for i, col in enumerate(podium_cols):
    driver = df.iloc[i]
    with col:
        pos_label = ["🥇 P1", "🥈 P2", "🥉 P3"][i]
        st.metric(
            label=pos_label,
            value=driver["name"],
            delta=f"{driver['p_podium']:.1%} podium probability",
        )
        st.caption(
            f"{driver['team']} · P(win) = {driver['p_win']:.1%} · "
            f"E[pos] = {driver['e_pos']:.1f}"
        )

# =====================================================================
# FULL GRID TABLE + BAR CHART
# =====================================================================
st.header("📊 Full Grid Probabilities")

col_table, col_chart = st.columns([1, 1])

with col_table:
    display_df = df[["rank", "name", "team", "p_win", "p_podium", "p_points", "p_dnf", "e_pos"]].copy()
    display_df.columns = ["Rank", "Driver", "Team", "P(Win)", "P(Podium)", "P(Points)", "P(DNF)", "E[Pos]"]
    for c in ["P(Win)", "P(Podium)", "P(Points)", "P(DNF)"]:
        display_df[c] = display_df[c].map("{:.1%}".format)
    display_df["E[Pos]"] = display_df["E[Pos]"].map("{:.1f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

with col_chart:
    fig_podium = go.Figure()
    fig_podium.add_trace(go.Bar(
        y=df["abbreviation"], x=df["p_win"],
        name="P(Win)", orientation="h",
        marker_color=[d["team_color"] for _, d in df.iterrows()],
        opacity=1.0,
        text=df["p_win"].map("{:.1%}".format), textposition="auto",
    ))
    fig_podium.add_trace(go.Bar(
        y=df["abbreviation"], x=df["p_podium"] - df["p_win"],
        name="P(P2-P3)", orientation="h",
        marker_color=[d["team_color"] for _, d in df.iterrows()],
        opacity=0.5,
        text=(df["p_podium"] - df["p_win"]).map("{:.1%}".format), textposition="auto",
    ))
    fig_podium.update_layout(
        barmode="stack", title="Win & Podium Probabilities",
        xaxis_title="Probability", yaxis=dict(autorange="reversed"),
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=10, t=60, b=40),
    )
    st.plotly_chart(fig_podium, use_container_width=True)

# =====================================================================
# DRIVER DEEP-DIVE
# =====================================================================
st.header("🔍 Driver Deep-Dive")

selected_drivers = st.multiselect(
    "Select drivers to compare",
    options=df["name"].tolist(),
    default=df["name"].tolist()[:5],
)

if selected_drivers:
    sel_df = df[df["name"].isin(selected_drivers)]
    col_dist, col_box = st.columns(2)

    with col_dist:
        fig_dist = go.Figure()
        for _, driver in sel_df.iterrows():
            pos_dist = driver["position_distribution"]
            labels = ["DNF"] + [f"P{i}" for i in range(1, len(pos_dist))]
            fig_dist.add_trace(go.Bar(
                x=labels, y=pos_dist,
                name=f"{driver['abbreviation']} ({driver['team']})",
                marker_color=driver["team_color"], opacity=0.7,
            ))
        fig_dist.update_layout(
            title="Position Probability Distribution",
            xaxis_title="Finishing Position", yaxis_title="Probability",
            barmode="group", height=450,
            margin=dict(l=0, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_box:
        fig_range = go.Figure()
        for _, driver in sel_df.iterrows():
            p = driver["percentiles"]
            fig_range.add_trace(go.Box(
                name=driver["abbreviation"],
                q1=[p["p25"]], median=[p["p50"]], q3=[p["p75"]],
                lowerfence=[p["p5"]], upperfence=[p["p95"]],
                marker_color=driver["team_color"],
                fillcolor=driver["team_color"],
                opacity=0.6, boxpoints=False,
            ))
        fig_range.update_layout(
            title="Finishing Position Range (5th-95th percentile)",
            yaxis_title="Position", yaxis=dict(autorange="reversed"),
            height=450, showlegend=False,
            margin=dict(l=0, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_range, use_container_width=True)

# =====================================================================
# MODEL COMPARISON (all stage models)
# =====================================================================
st.header("⚖️ Model Comparison")

# Build merged dataframe with all stage models
dfs_by_stage = {}
for sk in STAGE_KEYS:
    sdf = get_drivers_df(sk)
    dfs_by_stage[sk] = sdf

# Use first stage as base for merge
base_sk = STAGE_KEYS[0]
merged = dfs_by_stage[base_sk][["name", "team", "team_color"]].copy()
for sk in STAGE_KEYS:
    sdf = dfs_by_stage[sk]
    merged = merged.merge(
        sdf[["name", "p_win", "p_podium", "e_pos"]].rename(
            columns={c: f"{c}_{sk}" for c in ["p_win", "p_podium", "e_pos"]}
        ),
        on="name",
    )

col_comp1, col_comp2 = st.columns(2)

# Color palette for model comparison bars
COMPARISON_COLORS = {
    "stage6": "#4ECDC4",
    "stage8": "#FFD93D",
    "stage9": "#FF6B6B",
    "stage10": "#A78BFA",
}

with col_comp1:
    # Sort by composite or last stage for consistent ordering
    sort_col = f"p_win_{STAGE_KEYS[-1]}"
    sort_order = merged.sort_values(sort_col, ascending=True)

    fig_comp = go.Figure()
    for sk in STAGE_KEYS:
        short_label = MODEL_LABELS[sk].split(":")[0].strip()
        color = COMPARISON_COLORS.get(sk, "#888888")
        fig_comp.add_trace(go.Bar(
            y=sort_order["name"], x=sort_order[f"p_win_{sk}"],
            name=short_label, orientation="h",
            marker_color=color, opacity=0.8,
        ))
    fig_comp.update_layout(
        title="P(Win) — All Stage Models",
        xaxis_title="Probability", barmode="group",
        height=650, margin=dict(l=0, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col_comp2:
    # Expected position: scatter plot of first vs last stage, colored by team
    if len(STAGE_KEYS) >= 2:
        sk_x, sk_y = STAGE_KEYS[0], STAGE_KEYS[-1]
        label_x = MODEL_LABELS[sk_x].split(":")[0].strip()
        label_y = MODEL_LABELS[sk_y].split(":")[0].strip()

        fig_epos = go.Figure()
        fig_epos.add_trace(go.Scatter(
            x=merged[f"e_pos_{sk_x}"], y=merged[f"e_pos_{sk_y}"],
            mode="markers+text",
            text=merged["name"].str.split().str[-1],
            textposition="top center",
            marker=dict(size=12, color=merged["team_color"],
                        line=dict(width=1, color="white")),
            hovertemplate="%{text}<br>" + label_x + ": %{x:.1f}<br>" + label_y + ": %{y:.1f}<extra></extra>",
        ))
        max_pos = max(merged[f"e_pos_{sk_x}"].max(), merged[f"e_pos_{sk_y}"].max()) + 1
        fig_epos.add_trace(go.Scatter(
            x=[1, max_pos], y=[1, max_pos],
            mode="lines", line=dict(dash="dash", color="gray"),
            showlegend=False,
        ))
        fig_epos.update_layout(
            title=f"Expected Position — {label_x} vs {label_y}",
            xaxis_title=f"{label_x} E[Pos]", yaxis_title=f"{label_y} E[Pos]",
            xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"),
            height=650, margin=dict(l=0, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_epos, use_container_width=True)

# =====================================================================
# MODEL AGREEMENT / DISAGREEMENT TABLE
# =====================================================================
st.subheader("Model Agreement")

agree_data = []
for _, row in merged.iterrows():
    wins = {sk: row[f"p_win_{sk}"] for sk in STAGE_KEYS}
    pods = {sk: row[f"p_podium_{sk}"] for sk in STAGE_KEYS}
    epos = {sk: row[f"e_pos_{sk}"] for sk in STAGE_KEYS}
    win_vals = list(wins.values())
    pod_vals = list(pods.values())
    epos_vals = list(epos.values())
    agree_data.append({
        "Driver": row["name"],
        "Team": row["team"],
        "P(Win) Range": f"{min(win_vals):.1%} – {max(win_vals):.1%}",
        "P(Podium) Range": f"{min(pod_vals):.1%} – {max(pod_vals):.1%}",
        "E[Pos] Range": f"{min(epos_vals):.1f} – {max(epos_vals):.1f}",
        "Win Spread": max(win_vals) - min(win_vals),
        "Podium Spread": max(pod_vals) - min(pod_vals),
    })

agree_df = pd.DataFrame(agree_data).sort_values("Podium Spread", ascending=False)
st.markdown("Drivers sorted by **model disagreement** (largest podium probability spread first):")
st.dataframe(
    agree_df[["Driver", "Team", "P(Win) Range", "P(Podium) Range", "E[Pos] Range"]],
    use_container_width=True, hide_index=True,
)

# =====================================================================
# CALIBRATION CHECK (conditional — only if data has calibration info)
# =====================================================================
cal = data.get("calibration") or data.get("calibration_2025")
if cal:
    st.header("🎯 Calibration Check")
    actual_top3 = cal.get("top3", [])
    actual_top10 = cal.get("top10", [])
    cal_race = cal.get("race", "prior season")
    top3_str = ", ".join(f"**{d}**" for d in actual_top3)
    st.markdown(
        f"How well would our models have predicted the **{cal_race}**? "
        f"Actual top 3: {top3_str}."
    )

    cal_model_keys = MODEL_KEYS
    n_cal = len(cal_model_keys)
    cal_cols = st.columns(min(n_cal, 4))

    for i, mk in enumerate(cal_model_keys):
        mdf = get_drivers_df(mk)
        pred_top3 = mdf["name"].tolist()[:3]
        correct = sum(1 for d in pred_top3 if d in actual_top3)
        in_top10 = sum(1 for d in mdf["name"].tolist()[:10] if d in actual_top10)
        short_label = MODEL_LABELS[mk].split(":")[0].strip() if ":" in MODEL_LABELS[mk] else MODEL_LABELS[mk]
        with cal_cols[i % len(cal_cols)]:
            st.subheader(short_label)
            st.metric("Correct in Top 3", f"{correct}/3")
            st.metric("Top 10 Overlap", f"{in_top10}/10")
            for j, name in enumerate(pred_top3):
                marker = "✅" if name in actual_top3 else "❌"
                st.markdown(f"P{j+1}: {marker} {name}")

# =====================================================================
# CONSTRUCTOR ANALYSIS
# =====================================================================
st.header("🏎️ Constructor Analysis")

team_stats = []
for team in df["team"].unique():
    team_drivers = df[df["team"] == team]
    team_stats.append({
        "Team": team,
        "Color": team_drivers.iloc[0]["team_color"],
        "Best Driver": team_drivers.iloc[0]["name"],
        "P(Any Podium)": 1 - np.prod(1 - team_drivers["p_podium"].values),
        "P(Any Win)": 1 - np.prod(1 - team_drivers["p_win"].values),
        "Best E[Pos]": team_drivers["e_pos"].min(),
        "Avg P(Points)": team_drivers["p_points"].mean(),
    })

team_df = pd.DataFrame(team_stats).sort_values("P(Any Podium)", ascending=False)

col_team1, col_team2 = st.columns(2)

with col_team1:
    fig_team = go.Figure()
    fig_team.add_trace(go.Bar(
        y=team_df["Team"], x=team_df["P(Any Podium)"],
        orientation="h", marker_color=team_df["Color"],
        text=team_df["P(Any Podium)"].map("{:.1%}".format), textposition="auto",
    ))
    fig_team.update_layout(
        title="Constructor P(Any Driver on Podium)",
        xaxis_title="Probability", yaxis=dict(autorange="reversed"),
        height=420, margin=dict(l=0, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_team, use_container_width=True)

with col_team2:
    fig_intra = go.Figure()
    for team in team_df["Team"].tolist():
        team_drivers = df[df["team"] == team].sort_values("p_podium", ascending=False)
        if len(team_drivers) == 2:
            d1, d2 = team_drivers.iloc[0], team_drivers.iloc[1]
            color = d1["team_color"]
            fig_intra.add_trace(go.Bar(
                y=[team], x=[d1["p_podium"]],
                orientation="h", marker_color=color, opacity=0.9,
                text=[f"{d1['abbreviation']} {d1['p_podium']:.1%}"],
                textposition="inside", showlegend=False,
            ))
            fig_intra.add_trace(go.Bar(
                y=[team], x=[-d2["p_podium"]],
                orientation="h", marker_color=color, opacity=0.5,
                text=[f"{d2['abbreviation']} {d2['p_podium']:.1%}"],
                textposition="inside", showlegend=False,
            ))
    fig_intra.update_layout(
        title="Intra-Team Battle (P(Podium))",
        xaxis_title="← Driver 2 | Driver 1 →",
        barmode="relative", height=420,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_intra, use_container_width=True)

# =====================================================================
# POSITION DISTRIBUTION HEATMAP
# =====================================================================
st.header("🎯 Position Distribution Heatmap")

selected_model_heatmap = st.radio(
    "Model for heatmap", MODEL_KEYS,
    format_func=lambda x: MODEL_LABELS[x],
    horizontal=True, key="heatmap_model",
)

hm_df = get_drivers_df(selected_model_heatmap)
pos_labels = ["DNF"] + [f"P{i}" for i in range(1, 21)]
heatmap_data = np.array([d["position_distribution"] for _, d in hm_df.iterrows()])

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data, x=pos_labels,
    y=[f"{d['abbreviation']} ({d['team']})" for _, d in hm_df.iterrows()],
    colorscale="YlOrRd", colorbar_title="Probability",
    hovertemplate="Driver: %{y}<br>Position: %{x}<br>P = %{z:.3f}<extra></extra>",
))
fig_heatmap.update_layout(
    title=f"Full Position Distribution — {MODEL_LABELS[selected_model_heatmap]}",
    xaxis_title="Finishing Position", height=700,
    margin=dict(l=0, r=10, t=40, b=40),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# =====================================================================
# METHODOLOGY & CAVEATS
# =====================================================================
with st.expander("ℹ️ Model Methodology & Caveats"):
    st.markdown("""
**Stage 6: Year-Weighted Constructor Priors** — Dirichlet-Multinomial Markov model with
recency-weighted constructor transition matrices. Geometric decay w=0.70 selected via
leave-last-year-out CV. Best calibrated model (avg LL/race = -59.0 across 4 eras).
Trained on 2015-2025 (11 years needed for stable kappa_c).

**Stage 8: Time-Varying Plackett-Luce** — Time-varying driver/constructor strengths
with exponential smoothing and Plackett-Luce ranking model. Best Spearman rho (0.800)
for ranking accuracy, but tends to overconcentrate probability on top performers.
MC sampling (3000 rankings) for position marginals.

**Stage 9: Bayesian State-Space** — Random walk on driver and constructor log-strengths
observed via Plackett-Luce rankings. MAP inference via L-BFGS-B with analytic gradients.
Best top-3 prediction accuracy (1.59/3 correct). Good calibration (LL/race = -60.0).

**Composite** — Equal-weight probability blend of all active stage models, renormalized.
Combines Stage 6's calibration, Stage 8's ranking, and Stage 9's top-3 accuracy.

**Caveats**:
- 2026 regulation changes are not modeled — predictions assume continuity from 2025.
- Cadillac (new team) has no constructor history; predictions rely on driver priors only.
- Models cannot anticipate pre-season testing results or development trajectory.
- Stage 8 and 9 overweight recent McLaren/Verstappen dominance.
- Stage 6 with broad training window (2015-2025) may underweight recent form.
    """)
