"""Utilities for loading and preparing project datasets.

This module merges carbon price data with MCO2 token metrics and appends the
average ESG score. The resulting dataset is saved to ``outputs/clean_dataset.csv``
relative to the project root.
"""

from pathlib import Path

import pandas as pd


# Resolve paths relative to the repository root regardless of the current
# working directory. ``BASE_DIR`` points one level above the ``src`` directory.
BASE_DIR = Path(__file__).resolve().parent.parent


# Load datasets -------------------------------------------------------------
carbon_prices = pd.read_csv(BASE_DIR / "data" / "carbon_prices.csv")
# Remove timezone information so that the date column matches the MCO2 token
# dataframe during merging.
carbon_prices["date"] = pd.to_datetime(carbon_prices["date"]).dt.tz_localize(None)

mco2_token = pd.read_csv(BASE_DIR / "data" / "mco2_token.csv")
mco2_token["date"] = pd.to_datetime(mco2_token["date"]).dt.tz_localize(None)

esg_scores = pd.read_csv(BASE_DIR / "data" / "esg_scores.csv")


# Merge datasets ------------------------------------------------------------
merged = pd.merge(carbon_prices, mco2_token, on="date", how="inner")


# Compute average ESG score -------------------------------------------------
# Some datasets label the total ESG column differently. Prefer the explicit
# "Total ESG Risk score" column but fall back to ``esg_total`` if necessary.
esg_column = (
    "Total ESG Risk score" if "Total ESG Risk score" in esg_scores.columns else "esg_total"
)
avg_esg_score = esg_scores[esg_column].mean()
merged["avg_esg_score"] = avg_esg_score


# Save clean dataset --------------------------------------------------------
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_dir / "clean_dataset.csv", index=False)

