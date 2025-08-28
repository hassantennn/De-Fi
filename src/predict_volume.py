import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    df_path = root / "outputs" / "clean_dataset.csv"
    df = pd.read_csv(df_path)
    if "volume_y" in df.columns:
        df = df.rename(columns={"volume_y": "volume"})
    X = df[["close", "circulating_supply", "holders"]]
    y = df["volume"]
    model = LinearRegression()
    model.fit(X, y)
    df["predicted_volume"] = model.predict(X)
    out_path = root / "outputs" / "predictions.csv"
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
