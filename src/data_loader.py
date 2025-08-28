import pandas as pd

def load_carbon_prices(path="../data/carbon_prices.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_mco2_token(path="../data/mco2_token.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_esg_scores(path="../data/esg_scores.csv"):
    df = pd.read_csv(path)
    return df
