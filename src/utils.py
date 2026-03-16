"""Shared helpers: colors, formatters, discount band logic."""

import pandas as pd

CATEGORY_COLORS = {
    "Furniture": "#636EFA",
    "Office Supplies": "#EF553B",
    "Technology": "#00CC96",
}

REGION_COLORS = {
    "East": "#AB63FA",
    "West": "#FFA15A",
    "Central": "#19D3F3",
    "South": "#FF6692",
}

SEGMENT_COLORS = {
    "Consumer": "#1F77B4",
    "Corporate": "#FF7F0E",
    "Home Office": "#2CA02C",
}


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def create_discount_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Discount Band' categorical column to the DataFrame."""
    df = df.copy()
    bins = [-0.001, 0.099, 0.199, 0.299, 1.0]
    labels = ["0–10%", "10–20%", "20–30%", "30%+"]
    df["Discount Band"] = pd.cut(df["Discount"], bins=bins, labels=labels)
    return df


PYSPARK_CODE = '''
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \\
    .appName("SalesEDA") \\
    .master("local[*]") \\
    .getOrCreate()

# Load data
sdf = spark.read.csv("data/Dataset_Superstore.csv",
                     header=True, inferSchema=True)

# Add profit margin column
sdf = sdf.withColumn(
    "Profit_Margin_Pct",
    (F.col("Profit") / F.col("Sales")) * 100
)

# --- Aggregation 1: Region × Category × Segment ---
agg_df = sdf.groupBy("Region", "Category", "Segment").agg(
    F.round(F.sum("Sales"), 2).alias("Total_Sales"),
    F.round(F.sum("Profit"), 2).alias("Total_Profit"),
    F.round(F.avg("Profit_Margin_Pct"), 2).alias("Avg_Profit_Margin_Pct"),
    F.count("Order ID").alias("Order_Count"),
).orderBy("Region", "Category", "Segment")

# --- Aggregation 2: Running total by month (Window function) ---
monthly_sdf = sdf.withColumn(
    "YearMonth", F.date_format(F.to_date("Order Date", "M/d/yyyy"), "yyyy-MM")
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

agg_df.show()
monthly_agg.show()

agg_df.toPandas().to_csv("outputs/spark_results.csv", index=False)
monthly_agg.toPandas().to_csv("outputs/spark_monthly.csv", index=False)

spark.stop()
'''
