"""PySpark aggregations with a transparent pandas fallback.

When PySpark is not installed (e.g. Streamlit Community Cloud), every function
falls back to an equivalent pandas implementation so the app keeps working.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# PySpark availability check
# ---------------------------------------------------------------------------
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    pass


def _get_spark_session():
    spark = (
        SparkSession.builder
        .appName("SalesEDA")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_aggregations(df_pandas: pd.DataFrame):
    """Return (agg_df, monthly_df, used_spark: bool).

    *agg_df*     — Region × Category × Segment totals
    *monthly_df* — monthly sales with running total (window function result)
    *used_spark* — True if PySpark was used, False if pandas fallback ran
    """
    if SPARK_AVAILABLE:
        try:
            agg_df, monthly_df = _spark_aggregations(df_pandas)
            return agg_df, monthly_df, True
        except Exception:
            pass  # fall through to pandas

    agg_df, monthly_df = _pandas_aggregations(df_pandas)
    return agg_df, monthly_df, False


def save_results(agg_df: pd.DataFrame, monthly_df: pd.DataFrame,
                 output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    agg_df.to_csv(os.path.join(output_dir, "spark_results.csv"), index=False)
    monthly_df.to_csv(os.path.join(output_dir, "spark_monthly.csv"), index=False)


# ---------------------------------------------------------------------------
# PySpark implementation
# ---------------------------------------------------------------------------

def _spark_aggregations(df_pandas: pd.DataFrame):
    spark = _get_spark_session()

    sdf = spark.createDataFrame(df_pandas.astype(str))  # avoid type issues

    # Re-cast numeric columns
    for col in ["Sales", "Profit", "Discount", "Quantity"]:
        sdf = sdf.withColumn(col, F.col(col).cast("double"))

    sdf = sdf.withColumn(
        "Profit_Margin_Pct",
        (F.col("Profit") / F.col("Sales")) * 100
    )

    # --- Aggregation 1: Region × Category × Segment ---
    agg_sdf = sdf.groupBy("Region", "Category", "Segment").agg(
        F.round(F.sum("Sales"), 2).alias("Total_Sales"),
        F.round(F.sum("Profit"), 2).alias("Total_Profit"),
        F.round(F.avg("Profit_Margin_Pct"), 2).alias("Avg_Profit_Margin_Pct"),
        F.count("Order ID").alias("Order_Count"),
    ).orderBy("Region", "Category", "Segment")

    # --- Aggregation 2: Monthly running total (window function) ---
    monthly_sdf = sdf.withColumn(
        "YearMonth",
        F.date_format(
            F.to_date(F.col("Order Date"), "M/d/yyyy"), "yyyy-MM"
        )
    )
    monthly_agg = monthly_sdf.groupBy("YearMonth").agg(
        F.round(F.sum("Sales"), 2).alias("Monthly_Sales"),
        F.round(F.sum("Profit"), 2).alias("Monthly_Profit"),
    )
    window_spec = Window.orderBy("YearMonth")
    monthly_agg = monthly_agg.withColumn(
        "Running_Total_Sales",
        F.round(F.sum("Monthly_Sales").over(window_spec), 2)
    ).orderBy("YearMonth")

    agg_pandas = agg_sdf.toPandas()
    monthly_pandas = monthly_agg.toPandas()

    spark.stop()
    return agg_pandas, monthly_pandas


# ---------------------------------------------------------------------------
# Pandas fallback (identical semantics)
# ---------------------------------------------------------------------------

def _pandas_aggregations(df_pandas: pd.DataFrame):
    df = df_pandas.copy()
    df["Profit_Margin_Pct"] = (df["Profit"] / df["Sales"].replace(0, np.nan)) * 100

    agg_df = (
        df.groupby(["Region", "Category", "Segment"])
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Profit_Margin_Pct=("Profit_Margin_Pct", "mean"),
            Order_Count=("Order ID", "count"),
        )
        .round(2)
        .reset_index()
        .sort_values(["Region", "Category", "Segment"])
        .reset_index(drop=True)
    )

    monthly_df = (
        df.groupby("YearMonth")
        .agg(Monthly_Sales=("Sales", "sum"), Monthly_Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("YearMonth")
    )
    monthly_df["YearMonth"] = monthly_df["YearMonth"].astype(str)
    monthly_df["Monthly_Sales"] = monthly_df["Monthly_Sales"].round(2)
    monthly_df["Monthly_Profit"] = monthly_df["Monthly_Profit"].round(2)
    monthly_df["Running_Total_Sales"] = monthly_df["Monthly_Sales"].cumsum().round(2)
    monthly_df = monthly_df.reset_index(drop=True)

    return agg_df, monthly_df
