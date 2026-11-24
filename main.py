"""Predict wine quality from physicochemical measurements."""

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # The UCI client downloads the dataset when the script first runs.
    wine_quality = fetch_ucirepo(id=186)
    features = wine_quality.data.features
    target = wine_quality.data.targets.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Wine Quality Prediction")
    print(f"Rows used: {len(features)}")
    print(f"Features used: {len(features.columns)}")
    print(f"Mean absolute error: {mae:.2f} quality points")
    print(f"Root mean squared error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Example prediction for the first held-out wine sample.
    example = X_test.iloc[[0]]
    actual_quality = y_test.iloc[0]
    predicted_quality = model.predict(example)[0]
    print(f"\nExample actual score: {actual_quality}")
    print(f"Example predicted score: {predicted_quality:.1f}")


if __name__ == "__main__":
    main()
