"""All core analysis functions used by both the app and the notebook."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from src.utils import create_discount_bands


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def get_kpi_metrics(df: pd.DataFrame) -> dict:
    """Return the six headline KPIs."""
    order_values = df.groupby("Order ID")["Sales"].sum()
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()

    return {
        "total_revenue": total_sales,
        "total_profit": total_profit,
        "profit_margin_pct": (total_profit / total_sales) * 100,
        "total_orders": df["Order ID"].nunique(),
        "total_customers": df["Customer Name"].nunique(),
        "avg_order_value": order_values.mean(),
    }


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def get_yoy_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year revenue, profit and order count."""
    trend = (
        df.groupby("Year")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .reset_index()
    )
    trend["Profit Margin %"] = (trend["Profit"] / trend["Revenue"]) * 100
    trend["YoY Revenue Growth %"] = trend["Revenue"].pct_change() * 100
    return trend


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregated sales + profit (string YearMonth for plotting)."""
    monthly = (
        df.groupby("YearMonth")
        .agg(Monthly_Sales=("Sales", "sum"), Monthly_Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("YearMonth")
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    monthly["Running_Total_Sales"] = monthly["Monthly_Sales"].cumsum().round(2)
    return monthly


# ---------------------------------------------------------------------------
# Category / product analysis
# ---------------------------------------------------------------------------

def get_top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Top-n products by total profit."""
    return (
        df.groupby("Product Name")["Profit"]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={"Profit": "Total Profit"})
    )


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("Category")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "count"))
        .reset_index()
    )
    cat["Profit Margin %"] = (cat["Profit"] / cat["Sales"]) * 100
    return cat


def get_subcategory_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-category profit margins, sorted ascending (loss-makers first)."""
    sub = (
        df.groupby("Sub-Category")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    sub["Profit Margin %"] = (sub["Profit"] / sub["Sales"]) * 100
    sub = sub.sort_values("Profit Margin %").reset_index(drop=True)
    return sub


def get_top_bottom_subcategories(df: pd.DataFrame, n: int = 3):
    """Return (top-n, bottom-n) sub-categories by profit margin."""
    sub = get_subcategory_margin(df)
    return sub.nlargest(n, "Profit Margin %"), sub.nsmallest(n, "Profit Margin %")


# ---------------------------------------------------------------------------
# Regional / segment analysis
# ---------------------------------------------------------------------------

def get_region_analysis(df: pd.DataFrame) -> pd.DataFrame:
    region = (
        df.groupby("Region")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"), Customers=("Customer Name", "nunique"))
        .reset_index()
    )
    region["Profit Margin %"] = (region["Profit"] / region["Sales"]) * 100
    return region


def get_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    seg = (
        df.groupby("Segment")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .reset_index()
    )
    seg["Avg Order Value"] = seg["Sales"] / seg["Orders"]
    seg["Profit Margin %"] = (seg["Profit"] / seg["Sales"]) * 100
    return seg


# ---------------------------------------------------------------------------
# Business Intelligence metrics
# ---------------------------------------------------------------------------

def get_bi_metrics(df: pd.DataFrame) -> dict:
    """Key business findings surfaced as a structured dict."""
    sub = get_subcategory_margin(df)
    region = get_region_analysis(df)
    seg = get_segment_analysis(df)

    top3_sub = sub.nlargest(3, "Profit Margin %")[["Sub-Category", "Profit Margin %"]]
    bottom3_sub = sub.nsmallest(3, "Profit Margin %")[["Sub-Category", "Profit Margin %"]]
    best_region = region.loc[region["Profit Margin %"].idxmax()]
    best_segment = seg.loc[seg["Avg Order Value"].idxmax()]

    # Discount threshold where average profit per order turns negative
    disc_profit = (
        df.assign(DiscBucket=(df["Discount"] * 10).round() / 10)
        .groupby("DiscBucket")["Profit"]
        .mean()
        .reset_index()
        .sort_values("DiscBucket")
    )
    negatives = disc_profit[disc_profit["Profit"] < 0]
    breakeven_discount = float(negatives["DiscBucket"].min()) if len(negatives) > 0 else None

    return {
        "top3_subcategories": top3_sub,
        "bottom3_subcategories": bottom3_sub,
        "best_region_name": best_region["Region"],
        "best_region_margin": best_region["Profit Margin %"],
        "best_segment_name": best_segment["Segment"],
        "best_segment_aov": best_segment["Avg Order Value"],
        "breakeven_discount": breakeven_discount,
    }


# ---------------------------------------------------------------------------
# Discount analysis
# ---------------------------------------------------------------------------

def get_discount_analysis(df: pd.DataFrame):
    """Return (band_summary_df, breakeven_discount, df_with_bands)."""
    df_bands = create_discount_bands(df)

    bands = (
        df_bands.groupby("Discount Band", observed=True)
        .agg(
            Avg_Profit_Margin=("Profit Margin", "mean"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Sales", "count"),
        )
        .reset_index()
    )

    # Granular breakeven
    disc_profit = (
        df.assign(DiscBucket=(df["Discount"] * 10).round() / 10)
        .groupby("DiscBucket")["Profit"]
        .mean()
        .reset_index()
        .sort_values("DiscBucket")
    )
    negatives = disc_profit[disc_profit["Profit"] < 0]
    breakeven = float(negatives["DiscBucket"].min()) if len(negatives) > 0 else None

    return bands, breakeven, df_bands


# ---------------------------------------------------------------------------
# Advanced analytics
# ---------------------------------------------------------------------------

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Sales", "Profit", "Discount", "Quantity"]].corr()


def get_rfm_lite(df: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """Identify top-20% customers by total revenue (RFM-lite, no dates)."""
    cust = (
        df.groupby("Customer Name")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
        .reset_index(drop=True)
    )
    n_top = max(1, int(len(cust) * top_pct))
    cust["Tier"] = "Bottom 80%"
    cust.loc[:n_top - 1, "Tier"] = "Top 20%"
    cust["Revenue Contribution %"] = (cust["Sales"] / cust["Sales"].sum()) * 100
    cust["Cumulative Revenue %"] = cust["Revenue Contribution %"].cumsum()
    return cust


def get_shipping_analysis(df: pd.DataFrame) -> pd.DataFrame:
    ship = (
        df.groupby("Ship Mode")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "count"), Avg_Days=("Days to Ship", "mean"))
        .reset_index()
    )
    ship["Profit Margin %"] = (ship["Profit"] / ship["Sales"]) * 100
    return ship


# ---------------------------------------------------------------------------
# Predictive model
# ---------------------------------------------------------------------------

def run_regression(df: pd.DataFrame) -> dict:
    """Polynomial (degree-2) regression: predict Profit from Sales + Discount + Quantity.

    Plain linear regression scores R² ≈ −0.72 because the discount–profit
    relationship is non-linear (step-function collapse at ~30% discount).
    Adding degree-2 interaction terms captures this and lifts R² significantly.
    """
    features = ["Sales", "Discount", "Quantity"]
    X = df[features].copy()
    y = df["Profit"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Degree-2 polynomial features (Sales², Discount², Sales×Discount, etc.)
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr",   LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Extract named coefficients for display
    poly        = pipeline.named_steps["poly"]
    lr          = pipeline.named_steps["lr"]
    feat_names  = poly.get_feature_names_out(features)
    coeff_df    = (
        pd.DataFrame({"Feature": feat_names, "Coefficient": lr.coef_})
        .sort_values("Coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    # Also store the baseline linear R² for comparison
    lr_base = LinearRegression()
    lr_base.fit(X_train, y_train)
    r2_linear = r2_score(y_test, lr_base.predict(X_test))

    return {
        "model":       pipeline,
        "r2":          r2_score(y_test, y_pred),
        "r2_linear":   r2_linear,
        "mae":         mean_absolute_error(y_test, y_pred),
        "intercept":   float(lr.intercept_),
        "coefficients": coeff_df,
        "features":    features,
        "feat_names":  list(feat_names),
    }
