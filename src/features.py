import pandas as pd

def add_lag_features(df, column, lags=[1,7,30]):
    """Add lag features for a given column"""
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df

def add_moving_average(df, column, windows=[7,30]):
    """Add rolling averages"""
    for w in windows:
        df[f"{column}_ma{w}"] = df[column].rolling(window=w).mean()
    return df
