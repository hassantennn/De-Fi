import pandas as pd

carbon_prices = pd.read_csv("../data/carbon_prices.csv")
carbon_prices["date"] = pd.to_datetime(carbon_prices["date"])

mco2_token = pd.read_csv("../data/mco2_token.csv")
mco2_token["date"] = pd.to_datetime(mco2_token["date"])

esg_scores = pd.read_csv("../data/esg_scores.csv")
