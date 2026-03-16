"""Comprehensive Sales EDA Dashboard — 5-page Streamlit app."""

import os, sys, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_ROOT, "data", "Dataset_Superstore.csv")
sys.path.insert(0, _ROOT)

from src.data_loader import load_data
from src.eda_analysis import (
    get_kpi_metrics, get_yoy_trend, get_monthly_trend, get_top_products,
    get_category_summary, get_subcategory_margin, get_region_analysis,
    get_segment_analysis, get_bi_metrics, get_discount_analysis,
    get_correlation_matrix, get_rfm_lite, get_shipping_analysis, run_regression,
)
from src.spark_analysis import run_aggregations
from src.utils import PYSPARK_CODE

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.88rem;
    padding: 6px 0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64748b;
    font-size: 0.78rem;
}
[data-testid="stSidebarNavItems"] { gap: 2px; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 14px 0 0 14px;
}
.kpi-blue::before   { background: #3b82f6; }
.kpi-green::before  { background: #10b981; }
.kpi-amber::before  { background: #f59e0b; }
.kpi-purple::before { background: #8b5cf6; }
.kpi-rose::before   { background: #f43f5e; }
.kpi-cyan::before   { background: #06b6d4; }

.kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 6px;
    display: block;
}
.kpi-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 4px;
}
.kpi-value {
    color: #f1f5f9;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    color: #475569;
    font-size: 0.72rem;
    margin-top: 4px;
}

/* ── Page hero ── */
.page-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-hero::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
}
.page-hero h1 {
    color: #f1f5f9;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.page-hero p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
}

/* ── Section headers ── */
.section-title {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #334155, transparent);
    margin-left: 8px;
}

/* ── Info / Alert boxes ── */
.alert-box {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.5;
    border-left: 4px solid;
}
.alert-info    { background: #0c1a2e; border-color: #3b82f6; color: #93c5fd; }
.alert-success { background: #052e16; border-color: #10b981; color: #6ee7b7; }
.alert-warning { background: #2d1b00; border-color: #f59e0b; color: #fcd34d; }
.alert-danger  { background: #2d0a0a; border-color: #ef4444; color: #fca5a5; }

/* ── Finding cards ── */
.finding-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    height: 100%;
}
.finding-card .fc-icon { font-size: 1.4rem; margin-bottom: 8px; }
.finding-card .fc-title { color: #64748b; font-size: 0.72rem; font-weight: 600;
                          text-transform: uppercase; letter-spacing: .06em; }
.finding-card .fc-value { color: #f1f5f9; font-size: 1.1rem; font-weight: 700; margin: 4px 0; }
.finding-card .fc-desc  { color: #475569; font-size: 0.8rem; }

/* ── Predictor result box ── */
.predict-box {
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 16px;
    border: 1px solid;
    transition: all 0.3s;
}
.predict-profit { background: #052e16; border-color: #10b981; }
.predict-loss   { background: #2d0a0a; border-color: #ef4444; }
.predict-value  { font-size: 2rem; font-weight: 800; }
.predict-details { color: #64748b; font-size: 0.82rem; margin-top: 8px; }

/* ── Divider ── */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #334155, transparent);
    margin: 24px 0;
}

/* ── Streamlit metric override ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 16px 8px 8px;
    margin-bottom: 8px;
}
.sidebar-brand .brand-icon { font-size: 2.2rem; }
.sidebar-brand .brand-title {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 700;
    margin: 6px 0 2px;
}
.sidebar-brand .brand-sub { color: #475569; font-size: 0.72rem; }

/* ── Nav items — make labels visible ── */
[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    cursor: pointer;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio input:checked ~ label {
    background: #1e293b !important;
    color: #60a5fa !important;
}
/* Show the radio button text — do NOT hide it */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    display: block !important;
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}

/* ── Filter labels ── */
.stMultiSelect label { color: #94a3b8 !important; font-size: 0.8rem !important; }
.stMultiSelect [data-baseweb="tag"] {
    background: #1d4ed8 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared chart theme
# ─────────────────────────────────────────────────────────────────────────────
_CHART = dict(
    plot_bgcolor="#0a0f1e",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(t=30, b=20, l=10, r=10),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zerolinecolor="#334155"),
)

def _apply(fig, height=360, legend_top=False):
    fig.update_layout(**_CHART, height=height)
    if legend_top:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ))
    return fig


def _hero(icon, title, subtitle):
    st.markdown(f"""
    <div class="page-hero">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def _section(icon, title):
    st.markdown(f'<p class="section-title">{icon} {title}</p>', unsafe_allow_html=True)


def _divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


def _alert(msg, kind="info"):
    st.markdown(f'<div class="alert-box alert-{kind}">{msg}</div>',
                unsafe_allow_html=True)


def _finding(icon, title, value, desc, col):
    col.markdown(f"""
    <div class="finding-card">
        <div class="fc-icon">{icon}</div>
        <div class="fc-title">{title}</div>
        <div class="fc-value">{value}</div>
        <div class="fc-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def _kpi(icon, label, value, sub, accent, col):
    col.markdown(f"""
    <div class="kpi-card kpi-{accent}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def _load(): return load_data(DATA_PATH)

@st.cache_data(show_spinner="Running aggregations…")
def _run_agg(_df): return run_aggregations(_df)

@st.cache_data(show_spinner="Training model…")
def _run_reg(_df): return run_regression(_df)

df = _load()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">📊</div>
        <div class="brand-title">Sales Intelligence</div>
        <div class="brand-sub">Superstore · 2014–2017</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    page = st.radio(
        "page",
        options=[
            "📋  Executive Summary",
            "📦  Category & Regional",
            "💸  Discount Impact",
            "⚡  PySpark Results",
            "🤖  Predictive Model",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("**🔎 Filters**")

    years = sorted(df["Year"].unique())
    sel_years = st.multiselect("Year", years, default=years)

    categories = sorted(df["Category"].unique())
    sel_cats = st.multiselect("Category", categories, default=categories)

    regions = sorted(df["Region"].unique())
    sel_regions = st.multiselect("Region", regions, default=regions)

    fdf = df[
        df["Year"].isin(sel_years)
        & df["Category"].isin(sel_cats)
        & df["Region"].isin(sel_regions)
    ].copy()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Mini stats in sidebar
    st.markdown(f"""
    <div style="background:#0f172a;border-radius:10px;padding:12px 14px;border:1px solid #1e293b">
        <div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:8px">Active Dataset</div>
        <div style="color:#f1f5f9;font-size:1.2rem;font-weight:700">{len(fdf):,}</div>
        <div style="color:#475569;font-size:0.72rem">records selected</div>
        <div style="color:#475569;font-size:0.72rem;margin-top:6px">
            {df['Order Date'].min().strftime('%b %Y')} →
            {df['Order Date'].max().strftime('%b %Y')}
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
if "Executive Summary" in page:
    _hero("📋", "Executive Summary",
          f"Full dataset · {df['Order Date'].min().strftime('%b %d, %Y')} "
          f"→ {df['Order Date'].max().strftime('%b %d, %Y')} · 9,994 records")

    kpis = get_kpi_metrics(fdf)
    bi   = get_bi_metrics(fdf)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _kpi("💰", "Total Revenue",  f"${kpis['total_revenue']:,.0f}",       "All-time sales",       "blue",   c1)
    _kpi("📈", "Total Profit",   f"${kpis['total_profit']:,.0f}",        "Net earnings",         "green",  c2)
    _kpi("🎯", "Profit Margin",  f"{kpis['profit_margin_pct']:.1f}%",    "Revenue → profit",     "amber",  c3)
    _kpi("🛒", "Orders",         f"{kpis['total_orders']:,}",            "Unique order IDs",     "purple", c4)
    _kpi("👥", "Customers",      f"{kpis['total_customers']:,}",         "Unique customers",     "rose",   c5)
    _kpi("💳", "Avg Order Value", f"${kpis['avg_order_value']:,.2f}",    "Per order",            "cyan",   c6)

    _divider()

    # YoY trend + table
    col_l, col_r = st.columns([3, 2])
    with col_l:
        _section("📈", "Year-over-Year Revenue & Profit Trend")
        trend = get_yoy_trend(fdf)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["Year"], y=trend["Revenue"], name="Revenue",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=9, color="#3b82f6",
                        line=dict(color="#0a0f1e", width=2)),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        ))
        fig.add_trace(go.Scatter(
            x=trend["Year"], y=trend["Profit"], name="Profit",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=9, color="#10b981",
                        line=dict(color="#0a0f1e", width=2)),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.06)",
        ))
        _apply(fig, height=320, legend_top=True)
        fig.update_layout(xaxis_title="Year", yaxis_title="USD ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _section("📊", "YoY Summary")
        display = trend.copy()
        display["Revenue"]             = display["Revenue"].map("${:,.0f}".format)
        display["Profit"]              = display["Profit"].map("${:,.0f}".format)
        display["Profit Margin %"]     = display["Profit Margin %"].map("{:.1f}%".format)
        display["YoY Revenue Growth %"] = display["YoY Revenue Growth %"].map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
        )
        st.dataframe(display.set_index("Year"), use_container_width=True, height=270)

    _divider()

    # Top products + Key findings side by side
    col_l, col_r = st.columns([2, 3])

    with col_l:
        _section("🏆", "Top 5 Products by Profit")
        top5 = get_top_products(fdf, 5)
        fig = px.bar(
            top5, x="Total Profit", y="Product Name",
            orientation="h", color="Total Profit",
            color_continuous_scale=[[0, "#1d4ed8"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
            labels={"Total Profit": "Profit ($)", "Product Name": ""},
        )
        fig.update_traces(
            texttemplate="$%{x:,.0f}", textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        )
        _apply(fig, height=280)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _section("💡", "Key Business Findings")
        st.markdown("<br>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        _finding("🌍", "Best Region",
                 bi["best_region_name"],
                 f"{bi['best_region_margin']:.1f}% profit margin",
                 fc1)
        if bi["breakeven_discount"] is not None:
            _finding("⚠️", "Discount Danger Zone",
                     f"≥ {bi['breakeven_discount']*100:.0f}% discount",
                     "Profit turns negative here",
                     fc2)
        else:
            _finding("✅", "Discount Safe", "No breakeven found",
                     "All discount levels show positive profit", fc2)
        _finding("💳", "Best AOV Segment",
                 bi["best_segment_name"],
                 f"${bi['best_segment_aov']:,.2f} avg order value",
                 fc3)

        st.markdown("<br>", unsafe_allow_html=True)
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown("**📉 Loss-Making Sub-Categories**")
            for _, row in bi["bottom3_subcategories"].iterrows():
                color = "#ef4444"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 10px;background:#0f172a;border-radius:6px;'
                    f'margin:3px 0;border-left:3px solid {color}">'
                    f'<span style="color:#e2e8f0;font-size:0.83rem">{row["Sub-Category"]}</span>'
                    f'<span style="color:{color};font-weight:700;font-size:0.83rem">'
                    f'{row["Profit Margin %"]:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
        with sub2:
            st.markdown("**📈 Top Sub-Categories**")
            for _, row in bi["top3_subcategories"].iterrows():
                color = "#10b981"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 10px;background:#0f172a;border-radius:6px;'
                    f'margin:3px 0;border-left:3px solid {color}">'
                    f'<span style="color:#e2e8f0;font-size:0.83rem">{row["Sub-Category"]}</span>'
                    f'<span style="color:{color};font-weight:700;font-size:0.83rem">'
                    f'{row["Profit Margin %"]:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — CATEGORY & REGIONAL
# ─────────────────────────────────────────────────────────────────────────────
elif "Category & Regional" in page:
    _hero("📦", "Category & Regional Analysis",
          "Sales breakdown by product category, sub-category, region, and customer segment")

    cat_df    = get_category_summary(fdf)
    sub_df    = get_subcategory_margin(fdf)
    region_df = get_region_analysis(fdf)
    seg_df    = get_segment_analysis(fdf)

    # ── Category row ──
    _section("🗂️", "Sales by Category")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            cat_df, values="Sales", names="Category",
            color_discrete_map={
                "Furniture": "#636EFA",
                "Office Supplies": "#EF553B",
                "Technology": "#00CC96",
            },
            hole=0.55,
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
        )
        _apply(fig, height=300)
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05,
                        font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_sorted = cat_df.sort_values("Sales", ascending=True)
        fig = px.bar(
            cat_sorted, x="Sales", y="Category", orientation="h",
            color="Profit Margin %", color_continuous_scale="RdYlGn",
            labels={"Sales": "Revenue ($)"},
        )
        fig.update_traces(
            texttemplate="%{marker.color:.1f}%",
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        )
        _apply(fig, height=300)
        fig.update_coloraxes(colorbar=dict(
            tickfont=dict(color="#94a3b8"), title=dict(text="Margin %", font=dict(color="#94a3b8"))
        ))
        st.plotly_chart(fig, use_container_width=True)

    _divider()

    # ── Sub-category margin ──
    _section("📊", "Profit Margin by Sub-Category — loss-makers highlighted in red")
    sub_df["Color"] = sub_df["Profit Margin %"].apply(lambda x: "Loss" if x < 0 else "Profit")

    fig = px.bar(
        sub_df, x="Sub-Category", y="Profit Margin %",
        color="Color",
        color_discrete_map={"Loss": "#ef4444", "Profit": "#10b981"},
        labels={"Profit Margin %": "Margin (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1.5)
    fig.update_traces(
        texttemplate="%{y:.1f}%", textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    )
    _apply(fig, height=380)
    fig.update_layout(xaxis_tickangle=-40, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    _divider()

    # ── Region + Segment ──
    c1, c2 = st.columns(2)
    with c1:
        _section("🗺️", "Regional Performance")
        region_sorted = region_df.sort_values("Profit Margin %", ascending=True)
        fig = go.Figure(go.Bar(
            x=region_sorted["Profit Margin %"],
            y=region_sorted["Region"],
            orientation="h",
            marker=dict(
                color=region_sorted["Profit Margin %"],
                colorscale=[[0, "#1d4ed8"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
                line=dict(color="#0a0f1e", width=0),
            ),
            text=region_sorted["Profit Margin %"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        _apply(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        _section("👥", "Segment — Avg Order Value vs Profit Margin")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_df["Segment"], y=seg_df["Avg Order Value"],
            name="Avg Order Value ($)", marker_color="#3b82f6",
            marker_line=dict(width=0),
        ))
        fig.add_trace(go.Bar(
            x=seg_df["Segment"], y=seg_df["Profit Margin %"],
            name="Profit Margin (%)", marker_color="#10b981",
            marker_line=dict(width=0),
        ))
        _apply(fig, height=280, legend_top=True)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    # Segment table
    _divider()
    _section("📋", "Segment Detail")
    seg_display = seg_df.copy()
    seg_display["Sales"]           = seg_display["Sales"].map("${:,.0f}".format)
    seg_display["Profit"]          = seg_display["Profit"].map("${:,.0f}".format)
    seg_display["Avg Order Value"] = seg_display["Avg Order Value"].map("${:,.2f}".format)
    seg_display["Profit Margin %"] = seg_display["Profit Margin %"].map("{:.1f}%".format)
    st.dataframe(seg_display.set_index("Segment"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — DISCOUNT IMPACT
# ─────────────────────────────────────────────────────────────────────────────
elif "Discount Impact" in page:
    _hero("💸", "Discount Impact Analysis",
          "How discounting affects profitability — scatter analysis, band breakdown, customer segmentation")

    bands, breakeven, df_bands = get_discount_analysis(fdf)

    if breakeven is not None:
        _alert(
            f"<strong>⚠️ Profit Danger Zone Detected</strong> — "
            f"Average order profit turns <strong>negative at ≥ {breakeven*100:.0f}% discount</strong>. "
            f"The orange line on the scatter plot marks this exact threshold.",
            "warning",
        )
    else:
        _alert("✅ No discount threshold found where profit turns negative in the current filter.", "success")

    _divider()

    # ── Scatter ──
    _section("🔵", "Discount vs Profit — by Category")
    fig = px.scatter(
        fdf, x="Discount", y="Profit",
        color="Category",
        color_discrete_map={
            "Furniture": "#636EFA",
            "Office Supplies": "#EF553B",
            "Technology": "#00CC96",
        },
        opacity=0.5,
        labels={"Discount": "Discount Rate", "Profit": "Profit ($)"},
        hover_data=["Sub-Category", "Sales", "Quantity"],
    )
    if breakeven is not None:
        fig.add_vline(
            x=breakeven, line_dash="dash", line_color="#f59e0b", line_width=2,
            annotation_text=f"  Breakeven ≈ {breakeven*100:.0f}%",
            annotation_position="top right",
            annotation_font=dict(color="#f59e0b", size=12),
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)
    _apply(fig, height=420, legend_top=True)
    st.plotly_chart(fig, use_container_width=True)

    _divider()
    c1, c2 = st.columns(2)

    with c1:
        _section("📊", "Avg Profit Margin by Discount Band")
        band_colors = ["#10b981" if v >= 0 else "#ef4444" for v in bands["Avg_Profit_Margin"]]
        fig = go.Figure(go.Bar(
            x=bands["Discount Band"].astype(str),
            y=bands["Avg_Profit_Margin"],
            marker_color=band_colors,
            marker_line=dict(width=0),
            text=bands["Avg_Profit_Margin"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1.5)
        _apply(fig, height=320)
        fig.update_layout(yaxis_title="Avg Profit Margin (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        _section("📦", "Order Count by Discount Band")
        fig = px.bar(
            bands, x="Discount Band", y="Order_Count",
            color="Avg_Profit_Margin", color_continuous_scale="RdYlGn",
            labels={"Order_Count": "# Orders"},
            text=bands["Order_Count"],
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            marker_line=dict(width=0),
        )
        _apply(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    _divider()
    c1, c2 = st.columns(2)

    with c1:
        _section("🚚", "Shipping Mode vs Profit Margin")
        ship = get_shipping_analysis(fdf)
        ship_sorted = ship.sort_values("Profit Margin %", ascending=True)
        fig = go.Figure(go.Bar(
            x=ship_sorted["Profit Margin %"],
            y=ship_sorted["Ship Mode"],
            orientation="h",
            marker=dict(
                color=ship_sorted["Avg_Days"],
                colorscale="Viridis",
                colorbar=dict(title="Avg Days", tickfont=dict(color="#94a3b8")),
            ),
            text=ship_sorted["Profit Margin %"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        _apply(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        _section("🔗", "Correlation Heatmap")
        corr = get_correlation_matrix(fdf)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_traces(textfont=dict(size=13, color="#f1f5f9"))
        _apply(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

    _divider()
    _section("👑", "RFM-Lite — Top 20% Customers by Revenue")
    rfm     = get_rfm_lite(fdf)
    top_rfm = rfm[rfm["Tier"] == "Top 20%"]
    pct     = top_rfm["Sales"].sum() / rfm["Sales"].sum() * 100

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Top 20% customer count", f"{len(top_rfm):,}")
    mc2.metric("Revenue from top 20%",   f"${top_rfm['Sales'].sum():,.0f}")
    mc3.metric("Share of total revenue", f"{pct:.1f}%")
    mc4.metric("Bottom 80% revenue share", f"{100-pct:.1f}%")

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(
            rfm.groupby("Tier")["Sales"].sum().reset_index(),
            values="Sales", names="Tier",
            color="Tier",
            color_discrete_map={"Top 20%": "#f59e0b", "Bottom 80%": "#1e293b"},
            hole=0.6,
        )
        fig.update_traces(textfont=dict(color="#e2e8f0"))
        _apply(fig, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_customers = top_rfm.head(10)
        fig = px.bar(
            top_customers.sort_values("Sales", ascending=True),
            x="Sales", y="Customer Name", orientation="h",
            color="Sales",
            color_continuous_scale=[[0, "#78350f"], [1, "#f59e0b"]],
            labels={"Sales": "Total Revenue ($)"},
        )
        fig.update_traces(
            texttemplate="$%{x:,.0f}", textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
            marker_line=dict(width=0),
        )
        _apply(fig, height=300)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PYSPARK RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif "PySpark" in page:
    _hero("⚡", "PySpark Aggregation Results",
          "GROUP BY aggregations and window functions — computed with Apache Spark or pandas fallback")

    agg_df, monthly_df, used_spark = _run_agg(fdf)

    if used_spark:
        _alert("⚡ <strong>Apache Spark (local mode)</strong> — results computed with PySpark.", "success")
    else:
        _alert(
            "ℹ️ <strong>Pandas fallback</strong> — PySpark is not installed in this environment. "
            "Install <code>pyspark==3.5.0</code> and re-run to use Apache Spark. "
            "Results are <strong>identical</strong>.",
            "info",
        )

    _divider()

    # Summary stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aggregation rows",   f"{len(agg_df):,}")
    m2.metric("Regions",            f"{agg_df['Region'].nunique()}")
    m3.metric("Category × Segment", f"{agg_df['Category'].nunique() * agg_df['Segment'].nunique()}")
    m4.metric("Monthly data points", f"{len(monthly_df):,}")

    _divider()

    # Aggregation table
    _section("📋", "Region × Category × Segment Aggregation Table")
    st.dataframe(
        agg_df.style.background_gradient(subset=["Avg_Profit_Margin_Pct"], cmap="RdYlGn"),
        use_container_width=True, height=400,
    )
    st.download_button(
        "⬇️  Download Aggregation CSV",
        data=agg_df.to_csv(index=False),
        file_name="spark_results.csv",
        mime="text/csv",
    )

    _divider()

    # Running total chart
    _section("📈", "Monthly Sales + Running Total (PySpark Window Function)")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_df["YearMonth"], y=monthly_df["Monthly_Sales"],
        name="Monthly Sales", marker_color="#3b82f6", opacity=0.75,
        marker_line=dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x=monthly_df["YearMonth"], y=monthly_df["Running_Total_Sales"],
        name="Running Total", mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=5),
        yaxis="y2",
    ))
    _apply(fig, height=400, legend_top=True)
    fig.update_layout(
        yaxis=dict(title="Monthly Sales ($)", titlefont=dict(color="#3b82f6"),
                   gridcolor="#1e293b"),
        yaxis2=dict(title="Running Total ($)", titlefont=dict(color="#f59e0b"),
                    overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    _divider()

    # Monthly table + code side by side
    c1, c2 = st.columns([2, 3])
    with c1:
        _section("📊", "Monthly Breakdown Table")
        md = monthly_df.copy()
        for col in ["Monthly_Sales", "Monthly_Profit", "Running_Total_Sales"]:
            md[col] = md[col].map("${:,.0f}".format)
        st.dataframe(md, use_container_width=True, height=350)

    with c2:
        _section("💻", "PySpark Code — exact queries used")
        st.code(PYSPARK_CODE, language="python")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif "Predictive" in page:
    _hero("🤖", "Predictive Model — Profit Regression",
          "Linear Regression baseline: predict Profit from Sales + Discount + Quantity (scikit-learn)")

    reg = _run_reg(fdf)

    # Model metric cards
    c1, c2, c3, c4 = st.columns(4)
    _kpi("🎯", "Poly R² (degree-2)", f"{reg['r2']:.4f}",
         "Polynomial regression",     "green",  c1)
    _kpi("📉", "Linear R² (baseline)", f"{reg['r2_linear']:.4f}",
         "Plain linear — underfits",   "rose",   c2)
    _kpi("📏", "MAE",                f"${reg['mae']:,.2f}",
         "Avg prediction error",       "amber",  c3)
    _kpi("🧮", "Features (poly)",    f"{len(reg['feat_names'])}",
         "After degree-2 expansion",   "cyan",   c4)

    _divider()
    _alert(
        "📐 <strong>Why polynomial regression?</strong> "
        "The discount–profit relationship is non-linear — profit collapses sharply at ≥ 30% discount "
        "(see Page 3). A plain linear model scores R² ≈ −0.72 because it cannot capture this. "
        "Adding degree-2 interaction terms (e.g. <code>Sales × Discount</code>) "
        f"lifts R² to <strong>{reg['r2']:.4f}</strong>. "
        "For production, a Gradient Boosting model would perform even better.",
        "info",
    )

    _divider()
    c1, c2 = st.columns(2)

    with c1:
        _section("📊", "Top 10 Feature Coefficients by Magnitude")
        coeff = reg["coefficients"].head(10).copy()
        bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in coeff["Coefficient"]]
        fig = go.Figure(go.Bar(
            x=coeff["Coefficient"],
            y=coeff["Feature"],
            orientation="h",
            marker_color=bar_colors,
            marker_line=dict(width=0),
            text=coeff["Coefficient"].map("{:+.3f}".format),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1.5)
        _apply(fig, height=320)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<p style="color:#475569;font-size:0.8rem">Sorted by absolute magnitude · '
            'Positive = increases profit · Negative = decreases profit</p>',
            unsafe_allow_html=True,
        )

    with c2:
        _section("📋", "All Polynomial Coefficients")
        cd = reg["coefficients"].copy()
        cd["Coefficient"] = cd["Coefficient"].map("{:+.5f}".format)
        st.dataframe(cd.set_index("Feature"), use_container_width=True, height=320)

    _divider()

    # ── Interactive predictor ──
    _section("🎮", "Interactive Profit Predictor")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;margin-bottom:16px">'
        'Adjust the sliders to simulate a hypothetical order and get an instant profit prediction.</p>',
        unsafe_allow_html=True,
    )

    pc1, pc2, pc3 = st.columns(3)
    sales_val = pc1.slider("💰  Sales ($)",    10.0, 5000.0, float(fdf["Sales"].median()),   10.0)
    disc_pct  = pc2.slider("🏷️  Discount (%)", 0,    80,     20,                             5)
    qty_val   = pc3.slider("📦  Quantity",     1,    14,     int(fdf["Quantity"].median()),  1)
    disc_val  = disc_pct / 100

    X_pred = pd.DataFrame([[sales_val, disc_val, qty_val]],
                           columns=["Sales", "Discount", "Quantity"])
    predicted_profit = float(reg["model"].predict(X_pred)[0])
    margin_pct = (predicted_profit / sales_val) * 100 if sales_val > 0 else 0

    is_profit  = predicted_profit >= 0
    box_class  = "predict-profit" if is_profit else "predict-loss"
    val_color  = "#10b981" if is_profit else "#ef4444"
    icon       = "✅" if is_profit else "❌"
    verdict    = "PROFITABLE" if is_profit else "LOSS-MAKING"

    st.markdown(f"""
    <div class="predict-box {box_class}">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <div>
                <div style="color:#64748b;font-size:0.72rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:.08em">{verdict}</div>
                <div class="predict-value" style="color:{val_color}">
                    {icon} ${predicted_profit:,.2f}
                </div>
                <div style="color:#475569;font-size:0.8rem">Predicted Profit</div>
            </div>
            <div style="width:1px;height:60px;background:#334155"></div>
            <div>
                <div style="color:#64748b;font-size:0.72rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:.08em">Profit Margin</div>
                <div style="color:{val_color};font-size:1.6rem;font-weight:700">
                    {margin_pct:.1f}%
                </div>
            </div>
            <div style="margin-left:auto;text-align:right">
                <div class="predict-details">
                    Sales: <strong style="color:#e2e8f0">${sales_val:,.0f}</strong><br>
                    Discount: <strong style="color:#e2e8f0">{disc_pct}%</strong><br>
                    Quantity: <strong style="color:#e2e8f0">{qty_val}</strong>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:20px 0 8px;
            border-top:1px solid #1e293b;text-align:center">
    <span style="color:#334155;font-size:0.78rem">
        Built with&nbsp;
        <span style="color:#475569">Streamlit</span> ·
        <span style="color:#475569">Plotly</span> ·
        <span style="color:#475569">scikit-learn</span> ·
        <span style="color:#475569">PySpark</span> ·
        <span style="color:#475569">pandas</span>
        &nbsp;|&nbsp;
        Superstore Dataset · 9,994 records
        &nbsp;|&nbsp;
        <a href="https://github.com/Khushipatel27/sales-eda-dashboard"
           style="color:#3b82f6;text-decoration:none">
           github.com/Khushipatel27/sales-eda-dashboard
        </a>
    </span>
</div>""", unsafe_allow_html=True)
