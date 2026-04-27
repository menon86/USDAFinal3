"""
USDA Rural Development Web Analytics Dashboard — 2025
Team: Jakub Jasinski, Mert Kiroglu, Anvitha Mandhadi, Rohan Menon
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────── CONFIG ───────────────────────────────
st.set_page_config(
    page_title="USDA RD Web Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── STYLING ──────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f4f6f4; }

    .usda-header {
        background: linear-gradient(135deg, #1a3a2a 0%, #2e6b3e 60%, #4a9e5c 100%);
        padding: 1.6rem 2rem; border-radius: 12px;
        margin-bottom: 1.4rem; color: white;
    }
    .usda-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .usda-header p  { font-size: 0.92rem; opacity: 0.85; margin: 0.3rem 0 0; }

    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #2e6b3e; margin-bottom: 0.6rem;
    }
    .kpi-card .kpi-val { font-size: 1.8rem; font-weight: 700; color: #1a3a2a; line-height: 1.1; }
    .kpi-card .kpi-lbl { font-size: 0.74rem; color: #6c757d; text-transform: uppercase;
                          letter-spacing: .06em; margin-top: 0.2rem; }

    .geo-callout {
        background: #e8f5e9; border: 1px solid #81c784;
        border-radius: 8px; padding: 0.75rem 1.1rem;
        margin: 0.7rem 0; font-size: 0.88rem; color: #1b5e20;
    }
    .geo-callout strong { color: #2e7d32; }

    .persona-badge {
        display: inline-block; padding: 0.28rem 0.85rem;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
        margin-right: 0.35rem; margin-bottom: 0.35rem;
    }
    .p0 { background:#fff3e0; color:#e65100; }
    .p1 { background:#e3f2fd; color:#0d47a1; }
    .p2 { background:#f3e5f5; color:#4a148c; }
    .p3 { background:#e8f5e9; color:#1b5e20; }

    .zombie-box {
        background: #ffebee; border: 2px dashed #e53935;
        border-radius: 8px; padding: 0.65rem 1rem;
        font-size: 0.85rem; color: #b71c1c; margin-bottom: 0.8rem;
    }

    .insight-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.7rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-top: 4px solid #2e6b3e;
    }
    .insight-card h4 { margin: 0 0 0.4rem; color: #1a3a2a; font-size: 0.95rem; }
    .insight-card p  { margin: 0; font-size: 0.86rem; color: #444; line-height: 1.5; }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1a3a2a;
        border-bottom: 2px solid #2e6b3e;
        padding-bottom: 0.28rem; margin-bottom: 0.9rem;
    }

    .sim-result-box {
        background: #f1f8e9; border: 1.5px solid #66bb6a;
        border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.8rem;
    }
    .sim-result-box h4 { margin: 0 0 0.5rem; color: #1b5e20; }

    div[data-testid="stTab"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────── CONSTANTS ────────────────────────────
SESSION_THRESHOLD = 50_000
CLUSTER_FEATURES  = ["Total Views per session", "Total Bounce rate", "Total Average session duration"]
OPTIMAL_K         = 4
PAL               = ["#e65100", "#0d47a1", "#4a148c", "#1b5e20"]

PERSONA_NAMES = {
    0: ("🎯 The Efficiency Seeker",        "p0"),
    1: ("🧭 The Informed Navigator",        "p1"),
    2: ("⚡ The Friction-Stalled Visitor",  "p2"),
    3: ("🔍 The Deep Researcher",           "p3"),
}
PERSONA_RECS = {
    0: "Users locate information quickly with low friction. Prioritise CTA optimisation and A/B test hero content to accelerate conversion pathways.",
    1: "Healthy browsing pattern with balanced engagement. Introduce progressive disclosure panels and internal linking to deepen dwell time.",
    2: "HIGHEST PRIORITY — users arrive and exit immediately. Audit mobile load speeds, improve above-the-fold content clarity, and deploy targeted on-page guidance tools.",
    3: "Highly engaged power users consuming detailed content. Add structured FAQ/knowledge-base sections and downloadable PDF resources to fully satisfy intent.",
}
MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─────────────────────────────── DATA LOADING ─────────────────────────
@st.cache_data(show_spinner="Loading USDA dataset…")
def load_data():
    # Support both XLSX and XLSB formats
    data_file = None
    if os.path.exists("usda_dataset_raw.xlsb"):
        data_file = "usda_dataset_raw.xlsb"
    elif os.path.exists("usda_dataset_raw.xlsx"):
        data_file = "usda_dataset_raw.xlsx"
    else:
        st.error("❌ Data file not found. Expected 'usda_dataset_raw.xlsb' or 'usda_dataset_raw.xlsx'")
        st.stop()
    
    # Read XLSB or XLSX
    if data_file.endswith('.xlsb'):
        try:
            # For XLSB: use pyxlsb library
            import pyxlsb
            with pyxlsb.open_workbook(data_file) as wb:
                sheet = wb.sheets[0]  # Get first sheet (named 'data')
                rows = list(sheet.rows())
                # Convert to DataFrame
                df = pd.DataFrame([row for row in rows[1:]], columns=[cell.v for cell in rows[0]])
        except ImportError:
            st.error("""
            ❌ **pyxlsb library not found.**
            
            To read XLSB files, install it:
            ```bash
            pip install pyxlsb
            ```
            
            Or convert your file to XLSX using Excel/LibreOffice and upload that instead.
            """)
            st.stop()
    else:
        # Standard XLSX
        df = pd.read_excel(data_file, sheet_name="data")

    # Aggregate sessions per page and apply 50k threshold
    page_totals = (
        df.groupby("Page path and screen class")["Total Sessions"]
        .sum().reset_index()
        .rename(columns={"Total Sessions": "Agg Sessions"})
    )
    df = df.merge(page_totals, on="Page path and screen class", how="left")
    df = df[df["Agg Sessions"] >= SESSION_THRESHOLD].copy()

    # Monthly aggregation
    monthly = (
        df.groupby("Month")
        .agg(Sessions=("Total Sessions","sum"),
             Bounce=("Total Bounce rate","mean"),
             Duration=("Total Average session duration","mean"),
             Views=("Total Views per session","mean"))
        .reset_index().sort_values("Month")
    )
    monthly["Month Label"] = monthly["Month"].map(MONTH_MAP)

    # Device comparison
    device_agg = {}
    for dev in ["Desktop", "Mobile", "Tablet"]:
        device_agg[dev] = {
            "Sessions":      df[f"{dev} Sessions"].sum(),
            "Views/Session": df[f"{dev} Views per session"].mean(),
            "Avg Duration":  df[f"{dev} Average session duration"].mean(),
            "Bounce Rate":   df[f"{dev} Bounce rate"].mean(),
        }
    device_df = pd.DataFrame(device_agg).T.reset_index().rename(columns={"index": "Device"})

    # Page-level aggregation for clustering
    rd_agg = (
        df.groupby("Page path and screen class")
        .agg(**{
            "Total Views per session":        ("Total Views per session",        "mean"),
            "Total Bounce rate":              ("Total Bounce rate",              "mean"),
            "Total Average session duration": ("Total Average session duration", "mean"),
            "Total Sessions":                 ("Total Sessions",                 "sum"),
            "Page title":                     ("Page title",                     "first"),
        })
        .reset_index()
        .dropna(subset=CLUSTER_FEATURES)
    )

    # Fit scaler + KMeans
    scaler = StandardScaler()
    X = scaler.fit_transform(rd_agg[CLUSTER_FEATURES])
    km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    rd_agg["Cluster"] = km.fit_predict(X)
    rd_agg["Persona"] = rd_agg["Cluster"].map(lambda c: PERSONA_NAMES.get(c, (f"C{c}",""))[0])

    # Diagnostics for k=2..6
    ks, inertias, silhs = list(range(2, 7)), [], []
    for k in ks:
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = m.fit_predict(X)
        inertias.append(m.inertia_)
        silhs.append(silhouette_score(X, lbl))

    return df, monthly, device_df, rd_agg, scaler, km, inertias, silhs, ks


df, monthly, device_df, rd_agg, scaler, km_fitted, inertias, silhs, ks = load_data()

# ─────────────────────────────── HEADER ───────────────────────────────
st.markdown("""
<div class="usda-header">
  <h1>🌾 USDA Rural Development — Web Analytics Dashboard 2025</h1>
  <p>Executive Intelligence Platform &nbsp;·&nbsp; Pages ≥ 50,000 Aggregated Sessions &nbsp;·&nbsp; Domestic Traffic Focus</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="geo-callout">
  🌎 <strong>Geographic Context:</strong>
  <strong>99.57%</strong> of all recorded traffic originates from <strong>domestic (US) sources</strong>,
  consistent with midpoint findings. International traffic (Puerto Rico, Canada, India &amp; others)
  accounts for <strong>0.43%</strong> and is retained in the dataset but excluded from segment-level recommendations.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── TABS ─────────────────────────────────
tabs = st.tabs([
    "📊 Executive Briefing",
    "🔬 RD Segment Profiling",
    "🧪 Model Diagnostics",
    "🎛️ Page Simulator",
    "🏆 Strategic Insights",
    "📖 Manual",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE BRIEFING
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">System-Wide KPIs — High-Impact Pages (≥ 50k Sessions)</div>',
                unsafe_allow_html=True)

    total_sess   = df["Total Sessions"].sum()
    total_users  = df["Total users"].sum()
    avg_bounce   = df["Total Bounce rate"].mean()
    avg_duration = df["Total Average session duration"].mean()
    unique_pages = df["Page path and screen class"].nunique()
    total_events = df["Total Event count"].sum()

    kpi_cols = st.columns(6)
    kpi_data = [
        (f"{total_sess:,.0f}",   "Total Sessions"),
        (f"{total_users:,.0f}",  "Total Users"),
        (f"{total_events:,.0f}", "Total Events"),
        (f"{avg_bounce:.1%}",    "Avg Bounce Rate"),
        (f"{avg_duration:.0f}s", "Avg Session Duration"),
        (f"{unique_pages}",      "High-Impact Pages"),
    ]
    for col, (val, lbl) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Monthly Traffic Trend</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["Month Label"], y=monthly["Sessions"],
            marker_color="#2e6b3e", name="Sessions", opacity=0.82,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month Label"], y=monthly["Bounce"],
            mode="lines+markers", yaxis="y2",
            line=dict(color="#e53935", width=2.3, dash="dot"),
            marker=dict(size=6), name="Avg Bounce Rate",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month Label"], y=monthly["Duration"],
            mode="lines+markers", yaxis="y3",
            line=dict(color="#ff7043", width=2.3),
            marker=dict(size=6), name="Avg Duration (s)",
        ))
        fig.update_layout(
            yaxis =dict(title="Total Sessions", showgrid=True, gridcolor="#e8e8e8"),
            yaxis2=dict(title="Bounce Rate", overlaying="y", side="right",
                        tickformat=".0%", showgrid=False, anchor="free", position=0.93),
            yaxis3=dict(title="Duration (s)", overlaying="y", side="right",
                        showgrid=False, anchor="free", position=1.0),
            legend=dict(orientation="h", y=1.13, font_size=11),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=30, r=80), height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Day-of-Month Engagement Heatmap</div>',
                    unsafe_allow_html=True)
        day_pivot = (
            df.groupby(["Month","Day"])["Total Sessions"].sum()
            .reset_index()
            .pivot_table(index="Month", columns="Day", values="Total Sessions", aggfunc="sum")
            .fillna(0)
        )
        fig2 = px.imshow(
            day_pivot,
            color_continuous_scale=[[0,"#f1f8e9"],[0.4,"#66bb6a"],[1,"#1b5e20"]],
            labels=dict(x="Day of Month", y="Month", color="Sessions"),
            aspect="auto",
        )
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(t=20, b=20), height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📱 Device Friction Analysis</div>',
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1.1, 1, 0.9])

    with col_a:
        colors = {"Desktop":"#2e6b3e","Mobile":"#ff7043","Tablet":"#1565c0"}
        fig3 = go.Figure()
        for _, row in device_df.iterrows():
            dev = row["Device"]
            fig3.add_trace(go.Bar(
                name=dev,
                x=["Views/Session","Avg Duration (min)","Bounce Rate (%)"],
                y=[row["Views/Session"], row["Avg Duration"]/60, row["Bounce Rate"]*100],
                marker_color=colors.get(dev,"#888"),
            ))
        fig3.update_layout(
            barmode="group", title="Device Engagement Comparison",
            yaxis_title="Metric Value", plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.15), height=340, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("**⏱ The 3-Minute Vital Window (0–200s)**")
        st.caption("Behaviour is most volatile in the first 200s — the critical engagement decision zone.")
        window_data = []
        bins   = list(range(0, 210, 20))
        labels = [f"{i}-{i+20}s" for i in range(0, 200, 20)]
        for dev in ["Desktop","Mobile","Tablet"]:
            sub = df[df[f"{dev} Sessions"] > 0].copy()
            sub["bucket"] = pd.cut(
                sub[f"{dev} Average session duration"],
                bins=bins, labels=labels, include_lowest=True,
            )
            bk = sub.groupby("bucket", observed=True)[f"{dev} Sessions"].mean().reset_index()
            bk["Device"] = dev
            window_data.append(bk)
        wdf = pd.concat(window_data)
        wdf.columns = ["Bucket","Avg Sessions","Device"]
        fig4 = px.line(wdf, x="Bucket", y="Avg Sessions", color="Device",
                       color_discrete_map={"Desktop":"#2e6b3e","Mobile":"#ff7043","Tablet":"#1565c0"},
                       markers=True)
        fig4.add_vrect(x0="0-20s", x1="60-80s", fillcolor="#ffeb3b", opacity=0.15, line_width=0,
                       annotation_text="⚡ Peak flux", annotation_position="top left",
                       annotation_font_size=11)
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=320, margin=dict(t=20, b=10),
                           xaxis_tickangle=-40, legend=dict(font_size=11))
        st.plotly_chart(fig4, use_container_width=True)

    with col_c:
        st.markdown("**Device Traffic Share**")
        vals = [df["Desktop Sessions"].sum(), df["Mobile Sessions"].sum(),
                df["Tablet Sessions"].sum(),  df["Smart TV Sessions"].sum()]
        fig5 = go.Figure(go.Pie(
            labels=["Desktop","Mobile","Tablet","Smart TV"],
            values=vals, hole=0.58,
            marker_colors=["#2e6b3e","#ff7043","#1565c0","#8e24aa"],
            textinfo="label+percent", textfont_size=12,
        ))
        fig5.update_layout(showlegend=False, height=320,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Mobile vs Desktop — Engagement Delta</div>',
                unsafe_allow_html=True)
    mob_row  = device_df[device_df["Device"]=="Mobile"].iloc[0]
    desk_row = device_df[device_df["Device"]=="Desktop"].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Mobile Bounce Rate",    f"{mob_row['Bounce Rate']:.1%}",
              delta=f"{mob_row['Bounce Rate']-desk_row['Bounce Rate']:+.1%} vs Desktop",
              delta_color="inverse")
    d2.metric("Mobile Avg Duration",   f"{mob_row['Avg Duration']:.0f}s",
              delta=f"{mob_row['Avg Duration']-desk_row['Avg Duration']:+.0f}s vs Desktop",
              delta_color="normal")
    d3.metric("Mobile Views/Session",  f"{mob_row['Views/Session']:.2f}",
              delta=f"{mob_row['Views/Session']-desk_row['Views/Session']:+.2f} vs Desktop",
              delta_color="normal")
    d4.metric("Mobile Sessions Share",
              f"{df['Mobile Sessions'].sum()/df['Total Sessions'].sum():.1%}")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — RD SEGMENT PROFILING
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Rural Development Page Segments — K-Means Persona Profiles</div>',
                unsafe_allow_html=True)

    badge_html = "".join(
        f'<span class="persona-badge {cls}">{name}</span>'
        for _, (name, cls) in PERSONA_NAMES.items()
    )
    st.markdown(f'<div style="margin-bottom:1rem;">{badge_html}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown("**Persona Radar — Normalised Feature Profiles**")
        cats = ["Views/Session","Bounce Rate","Avg Duration","Log Sessions"]
        max_vals = [
            rd_agg["Total Views per session"].max(),
            100,
            rd_agg["Total Average session duration"].max() / 60,
            np.log1p(rd_agg["Total Sessions"].max()),
        ]
        fig_r = go.Figure()
        for c in range(OPTIMAL_K):
            sub = rd_agg[rd_agg["Cluster"]==c]
            raw = [
                sub["Total Views per session"].mean(),
                sub["Total Bounce rate"].mean() * 100,
                sub["Total Average session duration"].mean() / 60,
                np.log1p(sub["Total Sessions"].mean()),
            ]
            norm = [v/m*100 for v,m in zip(raw, max_vals)]
            norm += [norm[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=norm, theta=cats+[cats[0]],
                fill="toself", opacity=0.5,
                name=PERSONA_NAMES[c][0], line_color=PAL[c],
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            showlegend=True, height=430, paper_bgcolor="white",
            legend=dict(orientation="v", x=1.05, font_size=11),
            margin=dict(t=30, b=10, r=180),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.markdown("**Cluster Summary**")
        rows = []
        for c in range(OPTIMAL_K):
            sub = rd_agg[rd_agg["Cluster"]==c]
            rows.append({
                "Persona":        PERSONA_NAMES[c][0],
                "Pages":          len(sub),
                "Avg Sessions":   f"{sub['Total Sessions'].mean():,.0f}",
                "Avg Bounce":     f"{sub['Total Bounce rate'].mean():.1%}",
                "Avg Duration":   f"{sub['Total Average session duration'].mean():.0f}s",
                "Avg Views/Sess": f"{sub['Total Views per session'].mean():.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Strategic Recommendations**")
        for c in range(OPTIMAL_K):
            cls  = PERSONA_NAMES[c][1]
            name = PERSONA_NAMES[c][0]
            rec  = PERSONA_RECS[c]
            st.markdown(
                f'<span class="persona-badge {cls}">{name}</span>'
                f'<span style="font-size:0.84rem;color:#333;"> {rec}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")
    st.markdown('<div class="section-title">🧟 Friction Matrix — Zombie Session Detection</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="zombie-box">
    🧟 <strong>Zombie Session Zone</strong>: Pages with Avg Duration &gt; 1,000s <em>AND</em>
    Bounce Rate &gt; 50% indicate idle/abandoned sessions — browser tabs left open with no real engagement.
    These distort engagement metrics and require UX re-engagement intervention.
    </div>
    """, unsafe_allow_html=True)

    fig_fr = px.scatter(
        rd_agg,
        x="Total Average session duration", y="Total Bounce rate",
        color="Persona", size="Total Sessions", size_max=42,
        hover_data={"Page path and screen class": True, "Total Sessions": True},
        color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
        labels={
            "Total Average session duration": "Avg Session Duration (s)",
            "Total Bounce rate": "Bounce Rate",
        },
        title="Friction Matrix: Duration × Bounce Rate  (bubble size = session volume)",
    )
    max_dur = rd_agg["Total Average session duration"].max()
    fig_fr.add_shape(type="rect", x0=1000, x1=max_dur*1.05, y0=0.5, y1=1.02,
                     fillcolor="rgba(229,57,53,0.09)",
                     line=dict(color="#e53935", dash="dash", width=2))
    fig_fr.add_annotation(x=max_dur*0.88, y=0.97,
                           text="🧟 Zombie Session Zone", showarrow=False,
                           font=dict(color="#b71c1c", size=12))
    fig_fr.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=460,
        margin=dict(t=50, b=30),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee", tickformat=".0%"),
    )
    st.plotly_chart(fig_fr, use_container_width=True)

    # Bounce rate violin by cluster
    st.markdown('<div class="section-title">Bounce Rate Distribution by Persona</div>',
                unsafe_allow_html=True)
    fig_vio = go.Figure()
    for c in range(OPTIMAL_K):
        sub = rd_agg[rd_agg["Cluster"]==c]
        fig_vio.add_trace(go.Violin(
            y=sub["Total Bounce rate"], name=PERSONA_NAMES[c][0],
            box_visible=True, meanline_visible=True,
            fillcolor=PAL[c], line_color=PAL[c], opacity=0.55,
        ))
    fig_vio.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=340,
        yaxis=dict(title="Bounce Rate", tickformat=".0%"),
        showlegend=False, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_vio, use_container_width=True)

    # Top pages per cluster (expandable)
    st.markdown('<div class="section-title">Top Pages by Cluster</div>', unsafe_allow_html=True)
    for c in range(OPTIMAL_K):
        sub  = rd_agg[rd_agg["Cluster"]==c].nlargest(6, "Total Sessions")
        name = PERSONA_NAMES[c][0]
        with st.expander(f"{name} — Top 6 pages by sessions"):
            st.dataframe(
                sub[["Page path and screen class","Total Sessions",
                     "Total Bounce rate","Total Average session duration",
                     "Total Views per session"]]
                .rename(columns={
                    "Page path and screen class":     "Page Path",
                    "Total Sessions":                 "Sessions",
                    "Total Bounce rate":              "Bounce Rate",
                    "Total Average session duration": "Avg Duration (s)",
                    "Total Views per session":        "Views/Sess",
                }),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Model Technical Summary — K-Means Diagnostics</div>',
                unsafe_allow_html=True)
    st.info(
        f"**Optimal Cluster Count Selected: k = {OPTIMAL_K}**  \n"
        "Both the Elbow Method and Silhouette Score converge on this value, "
        "balancing intra-cluster cohesion with inter-cluster separation across "
        "three StandardScaler-normalised behavioural features."
    )

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**① Elbow Method — Inertia Plot**")
        fig_el = go.Figure()
        fig_el.add_trace(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color="#2e6b3e", width=2.5),
            marker=dict(size=9, color="#2e6b3e"), name="Inertia",
        ))
        fig_el.add_vline(x=OPTIMAL_K, line_dash="dash", line_color="#e53935", line_width=2,
                         annotation_text=f"Elbow @ k={OPTIMAL_K}",
                         annotation_position="top right", annotation_font_color="#e53935")
        fig_el.update_layout(
            xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (Within-Cluster SS)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickvals=ks), height=330, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_el, use_container_width=True)
        with st.expander("📐 What is Inertia?"):
            st.markdown("""
            **Inertia** (Within-Cluster Sum of Squares) measures cluster compactness.
            Lower = tighter clusters. The *elbow* marks diminishing returns — adding more
            clusters beyond this yields negligible compactness improvement.
            """)

    with d2:
        st.markdown("**② Silhouette Score — Cluster Separation**")
        bar_colors = ["#2e6b3e" if k==OPTIMAL_K else "#a5d6a7" for k in ks]
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(x=ks, y=silhs, marker_color=bar_colors,
                                  text=[f"{s:.3f}" for s in silhs], textposition="outside"))
        fig_sil.add_hline(y=max(silhs), line_dash="dot", line_color="#ff7043",
                           annotation_text=f"Peak @ k={ks[silhs.index(max(silhs))]}",
                           annotation_position="top right")
        fig_sil.update_layout(
            xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickvals=ks), height=330, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        with st.expander("📐 What is the Silhouette Score?"):
            st.markdown("""
            Range: **-1 to +1**.  
            - **+1** = Point is well inside its cluster, far from neighbours.  
            - **0** = Point sits on a cluster boundary.  
            - **-1** = Point may be misassigned.  
            A score ≥ 0.30 is acceptable for behavioural web analytics data.
            """)

    st.markdown("---")
    st.markdown('<div class="section-title">Feature Distributions by Cluster (z-score)</div>',
                unsafe_allow_html=True)
    X_sc = StandardScaler().fit_transform(rd_agg[CLUSTER_FEATURES])
    sc_df = pd.DataFrame(X_sc, columns=["Views/Session (z)","Bounce Rate (z)","Duration (z)"])
    sc_df["Cluster"] = rd_agg["Cluster"].values
    fig_box = go.Figure()
    for feat in ["Views/Session (z)","Bounce Rate (z)","Duration (z)"]:
        for c in range(OPTIMAL_K):
            sub = sc_df[sc_df["Cluster"]==c][feat]
            fig_box.add_trace(go.Box(
                y=sub, name=f"{PERSONA_NAMES[c][0][:22]} | {feat}",
                marker_color=PAL[c],
                showlegend=(feat=="Views/Session (z)"),
                legendgroup=str(c),
            ))
    fig_box.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(t=20, b=20),
        yaxis_title="Standardised Value (z-score)", boxmode="group",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Cluster Centroid Summary</div>', unsafe_allow_html=True)
    centroid_rows = []
    for c in range(OPTIMAL_K):
        sub = rd_agg[rd_agg["Cluster"]==c]
        centroid_rows.append({
            "Persona":              PERSONA_NAMES[c][0],
            "Views/Session (mean)": round(sub["Total Views per session"].mean(), 3),
            "Bounce Rate (mean)":   f"{sub['Total Bounce rate'].mean():.2%}",
            "Avg Duration (s)":     round(sub["Total Average session duration"].mean(), 1),
            "Total Pages":          len(sub),
            "Total Sessions":       f"{sub['Total Sessions'].sum():,.0f}",
        })
    st.dataframe(pd.DataFrame(centroid_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Pairwise Feature Scatter Matrix</div>',
                unsafe_allow_html=True)
    scatter_df = rd_agg[CLUSTER_FEATURES + ["Persona","Total Sessions"]].copy()
    scatter_df.columns = ["Views/Sess","Bounce Rate","Duration (s)","Persona","Sessions"]
    fig_sm = px.scatter_matrix(
        scatter_df,
        dimensions=["Views/Sess","Bounce Rate","Duration (s)"],
        color="Persona",
        color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
        size="Sessions", size_max=10, opacity=0.55,
    )
    fig_sm.update_traces(diagonal_visible=False)
    fig_sm.update_layout(height=480, paper_bgcolor="white", margin=dict(t=20, b=20))
    st.plotly_chart(fig_sm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — PAGE SIMULATOR
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🎛️ Page Performance Simulator</div>',
                unsafe_allow_html=True)
    st.caption(
        "Input hypothetical page metrics to receive an instant cluster assignment, persona mapping, "
        "benchmark comparison against real cluster averages, and a tailored strategic recommendation."
    )

    sim_l, sim_r = st.columns([1, 1.4])

    with sim_l:
        st.markdown("**Set Hypothetical Page Metrics**")
        dur_input    = st.slider("⏱ Avg Session Duration (seconds)", 0, 3000, 200, step=10)
        views_input  = st.slider("📄 Views per Session",              0.0, 15.0, 2.5, step=0.1)
        bounce_input = st.slider("↩ Bounce Rate",                    0.0,  1.0, 0.45, step=0.01,
                                 format="%.2f")
        st.markdown("---")
        run = st.button("🔍 Assign to Cluster", use_container_width=True, type="primary")

    with sim_r:
        if run:
            raw    = np.array([[views_input, bounce_input, dur_input]])
            scaled = scaler.transform(raw)
            cid    = int(km_fitted.predict(scaled)[0])
            pname, pcls = PERSONA_NAMES.get(cid, (f"Cluster {cid}", "p0"))
            rec    = PERSONA_RECS.get(cid, "")
            sub_c  = rd_agg[rd_agg["Cluster"]==cid]

            st.markdown(f"""
            <div class="sim-result-box">
              <h4>Assigned Persona</h4>
              <span class="persona-badge {pcls}" style="font-size:1rem;padding:0.45rem 1.1rem;">{pname}</span>
              <p style="margin-top:0.8rem;font-size:0.87rem;color:#1b5e20;">
                <strong>Recommendation:</strong> {rec}
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Benchmark vs Cluster Averages**")
            b1, b2, b3 = st.columns(3)
            b1.metric("Your Bounce Rate",  f"{bounce_input:.1%}",
                      delta=f"{bounce_input - sub_c['Total Bounce rate'].mean():+.1%} vs cluster",
                      delta_color="inverse")
            b2.metric("Your Duration",     f"{dur_input}s",
                      delta=f"{dur_input - sub_c['Total Average session duration'].mean():+.0f}s vs cluster",
                      delta_color="normal")
            b3.metric("Your Views/Sess",   f"{views_input:.1f}",
                      delta=f"{views_input - sub_c['Total Views per session'].mean():+.2f} vs cluster",
                      delta_color="normal")

            # Scatter: page positioned in cluster space
            fig_sim = px.scatter(
                rd_agg,
                x="Total Average session duration", y="Total Bounce rate",
                color="Persona", size="Total Sessions", size_max=28, opacity=0.45,
                color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
                labels={
                    "Total Average session duration": "Avg Duration (s)",
                    "Total Bounce rate": "Bounce Rate",
                },
                title="Your Page Positioned in Cluster Space",
            )
            fig_sim.add_trace(go.Scatter(
                x=[dur_input], y=[bounce_input], mode="markers",
                marker=dict(size=20, color="#ffeb3b", symbol="star",
                            line=dict(color="#f57f17", width=2.5)),
                name="⭐ Your Input",
            ))
            fig_sim.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=340, margin=dict(t=40, b=20),
                yaxis_tickformat=".0%", legend=dict(font_size=11),
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Spider: your profile vs cluster centroid
            st.markdown("**Your Metrics vs Cluster Centroid**")
            feats     = ["Views/Session","Bounce Rate (%)","Duration (min)"]
            your_vals = [views_input, bounce_input*100, dur_input/60]
            clus_vals = [
                sub_c["Total Views per session"].mean(),
                sub_c["Total Bounce rate"].mean()*100,
                sub_c["Total Average session duration"].mean()/60,
            ]
            maxv = [max(y,c)*1.15+0.001 for y,c in zip(your_vals, clus_vals)]
            yn   = [v/m*100 for v,m in zip(your_vals, maxv)] + [your_vals[0]/maxv[0]*100]
            cn   = [v/m*100 for v,m in zip(clus_vals, maxv)] + [clus_vals[0]/maxv[0]*100]
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatterpolar(
                r=yn, theta=feats+[feats[0]], fill="toself",
                name="Your Input", line_color="#f57f17",
                fillcolor="rgba(255,235,59,0.30)"))
            fig_sp.add_trace(go.Scatterpolar(
                r=cn, theta=feats+[feats[0]], fill="toself",
                name="Cluster Avg", line_color=PAL[cid],
                fillcolor="rgba(0,0,0,0.07)"))
            fig_sp.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                showlegend=True, height=320, paper_bgcolor="white",
                margin=dict(t=40, b=10, r=120),
            )
            st.plotly_chart(fig_sp, use_container_width=True)

        else:
            st.markdown("""
            <div style="padding:2.5rem;text-align:center;background:#f9fbe7;border-radius:10px;
                        border:1px dashed #aed581;color:#558b2f;">
              <div style="font-size:2.5rem;">🎛️</div>
              <p style="font-size:1rem;margin-top:0.8rem;">
                Adjust the sliders on the left and click<br>
                <strong>"Assign to Cluster"</strong> to get your results.
              </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Cluster Boundary Explorer</div>',
                unsafe_allow_html=True)
    st.caption("Per-cluster facet panels showing how Duration and Bounce Rate define each persona's territory.")
    fig_exp = px.scatter(
        rd_agg,
        x="Total Average session duration", y="Total Bounce rate",
        color="Persona", facet_col="Persona", facet_col_wrap=2,
        color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
        size="Total Sessions", size_max=22, opacity=0.65,
        labels={"Total Average session duration":"Duration (s)","Total Bounce rate":"Bounce Rate"},
        height=480,
    )
    fig_exp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=50, b=20), showlegend=False)
    fig_exp.for_each_yaxis(lambda a: a.update(tickformat=".0%"))
    st.plotly_chart(fig_exp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — STRATEGIC INSIGHTS
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏆 Strategic Insights — Executive Summary</div>',
                unsafe_allow_html=True)

    # Priority action matrix
    st.markdown("**Priority Action Matrix — Volume vs Friction**")
    st.caption("X = Avg Bounce Rate (friction level), Y = Total Sessions (traffic volume). Bubble size = Avg Duration.")

    fig_prio = px.scatter(
        rd_agg,
        x="Total Bounce rate", y="Total Sessions",
        color="Persona", size="Total Average session duration",
        size_max=35, opacity=0.72,
        hover_name="Page path and screen class",
        color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
        labels={"Total Bounce rate":"Bounce Rate (Friction)","Total Sessions":"Session Volume"},
        title="Priority Matrix — High-Volume + High-Friction = Immediate Action",
    )
    med_bounce = rd_agg["Total Bounce rate"].median()
    med_sess   = rd_agg["Total Sessions"].median()
    fig_prio.add_hline(y=med_sess, line_dash="dot", line_color="#9e9e9e",
                       annotation_text="Median Volume", annotation_position="bottom right")
    fig_prio.add_vline(x=med_bounce, line_dash="dot", line_color="#9e9e9e",
                       annotation_text="Median Bounce", annotation_position="top left")
    fig_prio.add_annotation(
        x=rd_agg["Total Bounce rate"].max()*0.88,
        y=rd_agg["Total Sessions"].max()*0.9,
        text="🚨 High Priority Zone", showarrow=False,
        font=dict(color="#b71c1c", size=13, family="Segoe UI"),
    )
    fig_prio.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=460, margin=dict(t=50, b=30),
        xaxis_tickformat=".0%",
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    )
    st.plotly_chart(fig_prio, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Top 10 Highest-Traffic Pages — Full Profile</div>',
                unsafe_allow_html=True)
    top10 = rd_agg.nlargest(10, "Total Sessions")[[
        "Page path and screen class","Persona","Total Sessions",
        "Total Bounce rate","Total Average session duration","Total Views per session",
    ]].rename(columns={
        "Page path and screen class":     "Page Path",
        "Total Sessions":                 "Sessions",
        "Total Bounce rate":              "Bounce Rate",
        "Total Average session duration": "Avg Duration (s)",
        "Total Views per session":        "Views/Session",
    })
    st.dataframe(
        top10.style
             .format({"Sessions":"{:,.0f}","Bounce Rate":"{:.1%}",
                      "Avg Duration (s)":"{:.0f}","Views/Session":"{:.2f}"})
             .background_gradient(subset=["Sessions"], cmap="Greens")
             .background_gradient(subset=["Bounce Rate"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Monthly Bounce Rate Trend — by Device</div>',
                unsafe_allow_html=True)
    monthly_dev = (
        df.groupby("Month")
        .agg(Desktop_Bounce=("Desktop Bounce rate","mean"),
             Mobile_Bounce =("Mobile Bounce rate", "mean"),
             Tablet_Bounce =("Tablet Bounce rate", "mean"))
        .reset_index()
    )
    monthly_dev["Month Label"] = monthly_dev["Month"].map(MONTH_MAP)
    fig_mb = go.Figure()
    for col, name, color in [
        ("Desktop_Bounce","Desktop","#2e6b3e"),
        ("Mobile_Bounce", "Mobile", "#ff7043"),
        ("Tablet_Bounce", "Tablet", "#1565c0"),
    ]:
        fig_mb.add_trace(go.Scatter(
            x=monthly_dev["Month Label"], y=monthly_dev[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.3), marker=dict(size=7),
        ))
    fig_mb.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=300,
        yaxis=dict(title="Avg Bounce Rate", tickformat=".0%",
                   showgrid=True, gridcolor="#eeeeee"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_mb, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Sessions vs Average Duration — All High-Impact Pages</div>',
                unsafe_allow_html=True)
    fig_sd = px.scatter(
        rd_agg,
        x="Total Average session duration", y="Total Views per session",
        color="Persona", size="Total Sessions", size_max=30,
        hover_name="Page path and screen class",
        color_discrete_map={PERSONA_NAMES[c][0]: PAL[c] for c in range(OPTIMAL_K)},
        labels={
            "Total Average session duration": "Avg Duration (s)",
            "Total Views per session": "Views per Session",
        },
        title="Engagement Depth Map — Duration vs Views/Session",
        opacity=0.75,
    )
    fig_sd.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=380, margin=dict(t=50, b=20))
    st.plotly_chart(fig_sd, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Executive Recommendations</div>',
                unsafe_allow_html=True)

    friction_pages = rd_agg[rd_agg["Cluster"]==2]["Total Sessions"].sum()
    research_pages = rd_agg[rd_agg["Cluster"]==3]["Total Sessions"].sum()
    mob_bounce     = device_df[device_df["Device"]=="Mobile"]["Bounce Rate"].values[0]
    desk_bounce    = device_df[device_df["Device"]=="Desktop"]["Bounce Rate"].values[0]

    insights = [
        ("🔴 Immediate — Address High-Friction Pages",
         f"Pages in the Friction-Stalled Visitor cluster account for "
         f"{friction_pages:,.0f} sessions with elevated bounce and minimal dwell time. "
         "Audit content relevance, above-the-fold clarity, and mobile load performance as "
         "the first remediation step. These pages represent the highest ROI for UX investment."),
        ("🟡 Short-Term — Close the Mobile Engagement Gap",
         f"Mobile visitors exhibit a bounce rate {mob_bounce - desk_bounce:+.1%} relative to Desktop. "
         "Prioritise responsive design audits and thumb-friendly navigation for the top 10 traffic pages. "
         "The 0–200s vital window reveals the decisive engagement moment — optimise first-screen content."),
        ("🟢 Medium-Term — Deepen Researcher Engagement",
         f"The Deep Researcher cluster drives {research_pages:,.0f} sessions of high-intent traffic. "
         "Introduce structured FAQ sections, programme comparison tools, and downloadable application "
         "guides to fully convert deep research intent into programme applications."),
        ("🔵 Ongoing — Eliminate Zombie Sessions",
         "Pages with Duration > 1,000s combined with Bounce Rate > 50% represent wasted server load "
         "and distorted analytics. Implement inactivity detection with re-engagement banners after "
         "5 minutes. Revisit these pages quarterly using this dashboard."),
        ("⚪ Data Governance — Revalidate Thresholds",
         f"The 50,000-session threshold currently isolates {rd_agg['Page path and screen class'].nunique()} "
         "high-impact pages. As traffic evolves through 2025, revalidate at mid-year to ensure "
         "emerging programme pages are captured in segment analysis."),
    ]
    for title, body in insights:
        st.markdown(f"""
        <div class="insight-card">
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — MANUAL
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">📖 User Manual — USDA RD Analytics Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("""
## Welcome, Executive User

This dashboard delivers a two-layer analytical assessment of USDA Rural Development web
effectiveness for 2025 — from system-wide KPIs through individual page persona profiling
and strategic action planning.

---

### 🗂️ Tab Overview

| Tab | Purpose | Primary Audience |
|-----|---------|-----------------|
| 📊 Executive Briefing | System-wide KPIs, monthly trends, device friction | Deputy Secretaries, Directors |
| 🔬 RD Segment Profiling | Cluster personas, friction matrix, page drill-down | Program Managers, UX Teams |
| 🧪 Model Diagnostics | Elbow method, silhouette scores, feature distributions | Data Scientists, Analysts |
| 🎛️ Page Simulator | What-if metric input → cluster assignment + recommendation | All levels |
| 🏆 Strategic Insights | Priority matrix, top pages, executive action plan | Leadership |
| 📖 Manual | This page | All levels |

---

### 📊 Tab 1 — Executive Briefing

**KPI Cards**: Six aggregate metrics for all pages meeting the 50,000-session threshold —
Sessions, Users, Events, Bounce Rate, Average Duration, and High-Impact Page Count.

**Monthly Trend (three-axis)**: Session volume (green bars), Bounce Rate (red dotted), and
Avg Duration (orange line) across all 12 months. Use this to align content campaigns with
peak traffic periods and detect seasonal anomalies.

**Day-of-Month Heatmap**: Intra-month traffic rhythms — darker green = higher sessions.
Useful for scheduling maintenance windows on low-traffic days.

**Device Friction Analysis (3 panels)**:
- Grouped bar chart comparing Desktop, Mobile, and Tablet across Views/Session,
  Duration, and Bounce Rate.
- The **3-Minute Vital Window** line chart isolates behaviour in the 0–200s decision zone,
  where users make their stay-or-leave decision.
- A traffic share donut showing device mix.

**Mobile vs Desktop Delta**: Four metric cards with directional delta indicators
highlighting the exact engagement gap between mobile and desktop users.

---

### 🔬 Tab 2 — RD Segment Profiling

**Persona Radar Chart**: Overlapping polygons showing the normalised behavioural profile
of each K-Means cluster. Larger polygon area on a dimension = stronger characteristic.

**Cluster Summary Table + Recommendations**: Side-by-side view of raw cluster statistics
and one-line strategic directives per persona.

**Friction Matrix**: Plotly scatter (X = Duration, Y = Bounce Rate, bubble = sessions).
The dashed red **Zombie Session Zone** (Duration > 1,000s + Bounce > 50%) flags pages
with idle/abandoned sessions requiring intervention.

**Bounce Rate Violin Chart**: Full distribution shape per cluster, revealing outliers
and variance — not just means.

**Top Pages Expanders**: Click any cluster to see the 6 highest-traffic pages with
full metric profiles.

---

### 🧪 Tab 3 — Model Diagnostics

**Elbow Plot**: Inertia vs k (2–6). Dashed line marks the selected k = 4.

**Silhouette Score Plot**: Bar chart of cluster separation quality. The tallest bar
(dark green) confirms the optimal k. Scores ≥ 0.30 are acceptable for behavioural analytics.

**Box Plots (z-score)**: StandardScaler-normalised feature distributions by cluster.
Non-overlapping boxes = well-separated clusters.

**Centroid Summary Table**: Mean feature values per cluster — the analytical fingerprint
of each persona, including total session count.

**Pairwise Scatter Matrix**: Feature-vs-feature scatter coloured by persona — reveals
multi-dimensional separation at a glance.

---

### 🎛️ Tab 4 — Page Simulator

1. Adjust the three sliders: **Avg Session Duration**, **Views/Session**, **Bounce Rate**.
2. Click **"Assign to Cluster"** to run the K-Means prediction on your inputs.
3. Results include:
   - **Assigned Persona** badge with colour coding.
   - **Benchmark delta metrics**: your input vs cluster averages.
   - **Scatter plot**: your page (⭐ yellow star) positioned against all real pages.
   - **Spider chart**: your metric profile overlaid on the cluster centroid.
4. **Cluster Boundary Explorer** (bottom): per-cluster facet panels for spatial orientation.

---

### 🏆 Tab 5 — Strategic Insights

**Priority Action Matrix**: Scatter plot of Volume (Y) vs Bounce/Friction (X).
Pages in the top-right quadrant (high sessions + high bounce) are the highest-priority
intervention targets — annotated with a 🚨 marker.

**Top 10 Pages Table**: Full-profile view of the ten highest-traffic pages with
colour-gradient formatting on Sessions (green intensity) and Bounce Rate (red-green scale).

**Monthly Bounce Rate Trend**: Device-split line chart tracking how bounce evolved
across the year — identifies seasonal friction spikes by device type.

**Engagement Depth Map**: Sessions vs Views/Session scatter, coloured by persona —
reveals which pages deliver substantive multi-page journeys.

**Executive Recommendations**: Five prioritised action cards (Immediate → Data Governance)
grounded directly in the cluster analysis, device metrics, and session thresholding logic.

---

### ⚙️ Methodology Reference

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Session Threshold | 50,000 | Isolates 30 high-impact pages; removes statistical noise |
| Algorithm | K-Means | Interpretable, scalable, reproducible |
| Optimal k | 4 | Validated by Elbow method + Silhouette Score |
| Feature Scaling | StandardScaler (z-score) | Prevents session duration from dominating distance calculations |
| Clustering Features | Views/Session, Bounce Rate, Avg Duration | Best behavioural discriminators in GA4 data |
| Geo Context | 99.57% US Domestic | International retained for completeness; excluded from recommendations |

---

### 👥 Project Team

| Name | Role |
|------|------|
| Jakub Jasinski | Lead Data Scientist |
| Mert Kiroglu | Machine Learning Engineer |
| Anvitha Mandhadi | Analytics & Visualisation |
| Rohan Menon | Strategic Insights & Reporting |

---
> **Data Source**: USDA Rural Development Google Analytics 4 Export · 2025  
> Re-run the app with an updated `usda_dataset_raw.xlsx` to refresh all metrics and cluster assignments automatically.
    """)


# ─────────────────────────────── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 USDA RD Analytics")
    st.markdown("**2025 Web Effectiveness Platform**")
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.markdown(f"- Rows (filtered): `{len(df):,}`")
    st.markdown(f"- Unique high-impact pages: `{df['Page path and screen class'].nunique()}`")
    st.markdown(f"- Session threshold: `≥ {SESSION_THRESHOLD:,}`")
    st.markdown(f"- Optimal clusters: `k = {OPTIMAL_K}`")
    st.markdown("---")
    st.markdown("**Geo Breakdown**")
    st.markdown("🇺🇸 **99.57% Domestic (US)**")
    st.markdown("🌐 0.43% International")
    st.markdown("---")
    st.markdown(
        "**Clustering Features**  \n"
        "- Views per Session  \n"
        "- Bounce Rate  \n"
        "- Avg Session Duration  \n\n"
        "*(StandardScaler normalised)*"
    )
    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("Jakub Jasinski  \nMert Kiroglu  \nAnvitha Mandhadi  \nRohan Menon")
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")
