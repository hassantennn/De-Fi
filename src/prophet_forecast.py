from pathlib import Path

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def main() -> None:
    """Forecast MCO2 trading volume using Prophet."""
    # Determine repository root and load the merged dataset
    root = Path(__file__).resolve().parent.parent
    df_path = root / "outputs" / "clean_dataset.csv"
    df = pd.read_csv(df_path)

    # Ensure the expected columns are present
    if "volume_y" in df.columns:
        df = df.rename(columns={"volume_y": "volume"})

    df = df[["date", "volume"]]
    df["date"] = pd.to_datetime(df["date"])

    # Prepare data for Prophet
    prophet_df = df.rename(columns={"date": "ds", "volume": "y"})

    # Fit Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Forecast 30 days into the future
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save forecast to the outputs directory
    forecast_path = root / "outputs" / "prophet_forecast.csv"
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        forecast_path, index=False
    )

    # Plot the forecast results
    model.plot(forecast)
    plt.title("MCO2 Trading Volume Forecast")
    plt.xlabel("Date")
    plt.ylabel("Trading Volume")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
