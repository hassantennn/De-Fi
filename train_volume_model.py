import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    df = pd.read_csv('outputs/clean_dataset.csv')
    if 'volume_y' in df.columns:
        df = df.rename(columns={'volume_y': 'volume'})
    X = df[['close', 'circulating_supply', 'holders']]
    y = df['volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")


if __name__ == '__main__':
    main()
