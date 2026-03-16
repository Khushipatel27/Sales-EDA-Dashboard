<div align="center">

# 📊 Comprehensive Sales EDA Dashboard
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/PySpark-3.5.0-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-5.20-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>


### An end-to-end sales analytics platform built with Streamlit, PySpark, and scikit-learn


<br/>


</div>

---

## ✨ Overview

A **portfolio-grade** sales analytics project that transforms raw Superstore transaction data into actionable business intelligence through:

- 🔥 **5-page interactive Streamlit dashboard** with dark-themed UI
- ⚡ **PySpark aggregations** (GROUP BY + window functions) with pandas fallback
- 🤖 **Polynomial Regression** model (R² = 0.72) for profit prediction
- 💡 **Business KPIs** — revenue, margin, discount breakeven, RFM segmentation
- 📓 **Jupyter notebook** with 5 analytical layers

---

## 📸 Dashboard Preview

| Page | Description |
|------|-------------|
| 📋 **Executive Summary** | 6 KPI cards · YoY trend · Top products · Key findings |
| 📦 **Category & Regional** | Category breakdown · Sub-category margins · Regional performance |
| 💸 **Discount Impact** | Scatter analysis · Band breakdown · Correlation heatmap · RFM-lite |
| ⚡ **PySpark Results** | Aggregation tables · Running total chart · Code viewer |
| 🤖 **Predictive Model** | Polynomial regression · Feature coefficients · Interactive predictor |

---

## 📊 Key Findings

> Computed from 9,994 Superstore orders (Jan 2014 – Dec 2017)

| Metric | Result |
|--------|--------|
| 💰 **Total Revenue** | **$2,297,201** |
| 📈 **Total Profit** | **$286,397** |
| 🎯 **Overall Profit Margin** | **12.5%** |
| 🛒 **Total Orders** | 5,009 |
| 👥 **Unique Customers** | 793 |
| ⚠️ **Discount Breakeven** | Profit turns negative at **≥ 30% discount** |
| 📉 **Loss-Making Sub-Categories** | Tables (−8.6%), Bookcases (−3.0%), Supplies (−2.5%) |
| 🏆 **Most Profitable Sub-Categories** | Labels (44.4%), Paper (43.4%), Envelopes (42.3%) |
| 🌍 **Best Region** | **West** — 14.9% profit margin |
| 💳 **Best Segment by AOV** | **Home Office** — $472.67 avg order value |
| 🤖 **Model R²** | **0.72** (Polynomial degree-2 regression) |

---

## 🏗️ Project Structure

```
📁 sales-eda-dashboard/
│
├── 📁 src/
│   ├── data_loader.py       # Load, validate & enrich data
│   ├── eda_analysis.py      # KPIs, trends, regression, RFM
│   ├── spark_analysis.py    # PySpark GROUP BY + window functions
│   └── utils.py             # Colour maps, formatters, PySpark code
│
├── 📁 data/
│   └── Dataset_Superstore.csv
│
├── 📁 notebooks/
│   └── 01_sales_eda.ipynb   # Full 5-layer EDA walkthrough
│
├── 📁 outputs/
│   └── figures/             # Chart exports
│
├── 📁 .streamlit/
│   └── config.toml          # Pinned dark theme
│
├── app.py                   # ← Streamlit entry point
├── requirements.txt
└── README.md
```

---

## ⚡ Tech Stack

| Layer | Tools |
|-------|-------|
| **Dashboard** | Streamlit 1.32 |
| **Data Engineering** | PySpark 3.5 · pandas 2.2 · numpy |
| **Visualisation** | Plotly Express / Graph Objects · matplotlib · seaborn |
| **Machine Learning** | scikit-learn — PolynomialFeatures · LinearRegression |
| **Notebook** | Jupyter · ipykernel |

---

## 🚀 Quick Start

### 1 · Clone

```bash
git clone https://github.com/Khushipatel27/sales-eda-dashboard.git
cd sales-eda-dashboard
```

### 2 · Create environment

```bash
conda create -n sales-eda python=3.10 -y
conda activate sales-eda
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Run the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔥 Enable PySpark (optional)

PySpark requires **Java 11+**. The app works fully without it (pandas fallback).

```bash
# Install Java via conda
conda install -c conda-forge openjdk=11 -y

# Uncomment pyspark in requirements.txt, then:
pip install pyspark==3.5.0
```

The **PySpark Results** page will show `⚡ Apache Spark (local mode)` when active.

---

## 📓 Run the Notebook

```bash
cd notebooks
jupyter notebook 01_sales_eda.ipynb
```

The notebook covers all 5 analytical layers and saves charts to `outputs/figures/`.

---

## 🗄️ Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Tableau Superstore Sample |
| **Records** | 9,994 orders |
| **Columns** | 21 |
| **Period** | January 2014 – December 2017 |
| **Geography** | United States · 4 regions · 49 states |

---

## 👩‍💻 Author

<div align="center">

**Khushi Patel**

[![GitHub](https://img.shields.io/badge/GitHub-Khushipatel27-181717?style=flat-square&logo=github)](https://github.com/Khushipatel27)

*Built for Summer 2026 internship applications — Data Analyst · Data Engineer · Data Scientist*

</div>

---

<div align="center">
<sub>⭐ Star this repo if you found it useful!</sub>
</div>
