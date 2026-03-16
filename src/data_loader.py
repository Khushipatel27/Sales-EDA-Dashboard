"""Load and validate the Superstore dataset."""

import os
import pandas as pd
import numpy as np

# Default path relative to repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(_HERE, "..", "data", "Dataset_Superstore.csv")


def load_data(path: str = None) -> pd.DataFrame:
    """Read CSV, parse dates, and derive analysis columns."""
    if path is None:
        path = DEFAULT_DATA_PATH

    df = pd.read_csv(path, encoding="latin1")

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Derived time columns
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["YearMonth"] = df["Order Date"].dt.to_period("M")

    # Derived metric columns
    df["Profit Margin"] = (df["Profit"] / df["Sales"].replace(0, np.nan)) * 100
    df["Profit Margin"] = df["Profit Margin"].fillna(0)

    # Days to ship
    df["Days to Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Return a dict of data-quality indicators."""
    missing = df.isnull().sum()
    return {
        "shape": df.shape,
        "missing_values": missing[missing > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": (df["Order Date"].min(), df["Order Date"].max()),
        "numeric_summary": df[["Sales", "Profit", "Discount", "Quantity"]].describe(),
    }


def get_data_summary(df: pd.DataFrame) -> dict:
    """Compute top-level KPI numbers from the full dataset."""
    order_totals = df.groupby("Order ID")["Sales"].sum()
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()

    return {
        "total_revenue": total_sales,
        "total_profit": total_profit,
        "profit_margin_pct": (total_profit / total_sales) * 100,
        "total_orders": df["Order ID"].nunique(),
        "total_customers": df["Customer Name"].nunique(),
        "avg_order_value": order_totals.mean(),
        "date_min": df["Order Date"].min(),
        "date_max": df["Order Date"].max(),
    }
