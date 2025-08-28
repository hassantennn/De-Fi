"""Exploratory analysis plots and statistics for carbon price and MCO2 token data.

This script loads the merged dataset produced by :mod:`load_dataframes` and
creates basic plots alongside a correlation matrix of the numerical columns.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Locate repository root so the script can be executed from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Read the merged dataset
merged = pd.read_csv(BASE_DIR / "outputs" / "clean_dataset.csv")
merged["date"] = pd.to_datetime(merged["date"])

# Rename columns for clarity
merged = merged.rename(
    columns={
        "volume_y": "mco2_volume",  # trading volume of MCO2 token
        "close": "carbon_price",   # carbon credit closing price
    }
)


def plot_mco2_volume_over_time() -> None:
    """Plot MCO2 trading volume over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(merged["date"], merged["mco2_volume"], label="Volume")
    plt.title("MCO2 Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("MCO2 Trading Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def scatter_carbon_price_vs_volume() -> None:
    """Scatter plot of carbon price against MCO2 trading volume."""
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["carbon_price"], merged["mco2_volume"])
    plt.title("Carbon Price vs MCO2 Trading Volume")
    plt.xlabel("Carbon Price")
    plt.ylabel("MCO2 Trading Volume")
    plt.tight_layout()
    plt.show()


def print_correlation_matrix() -> None:
    """Print the correlation matrix of all numerical columns."""
    corr_matrix = merged.select_dtypes(include="number").corr()
    print(corr_matrix)


if __name__ == "__main__":
    plot_mco2_volume_over_time()
    scatter_carbon_price_vs_volume()
    print_correlation_matrix()
