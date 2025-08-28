import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def main():
    df = pd.read_csv('outputs/clean_dataset.csv')
    if 'volume_y' in df.columns:
        df = df.rename(columns={'volume_y': 'volume'})
    X = df[['close', 'circulating_supply', 'holders']]
    y = df['volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Feature Importances:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")


if __name__ == '__main__':
    main()
