import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# MLflow configuration
# =========================
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file://" + os.path.abspath("mlruns")
)
mlflow.set_tracking_uri(TRACKING_URI)

# Aktifkan autolog (parameter, model, dsb)
mlflow.sklearn.autolog(log_models=False)

# =========================
# Data Loader
# =========================
def load_data(train_path, test_path, target):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if target not in train.columns or target not in test.columns:
        raise ValueError(f"Kolom target '{target}' tidak ditemukan")

    X_train = train.drop(columns=[target])
    y_train = train[target].values

    X_test = test.drop(columns=[target])
    y_test = test[target].values

    return X_train, X_test, y_train, y_test

# =========================
# Training & Logging
# =========================
def train_model(train_path, test_path, target):
    X_train, X_test, y_train, y_test = load_data(
        train_path, test_path, target
    )

    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Manual metrics (jelas untuk penilaian)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Simpan model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Run ID :", run.info.run_id)
        print(f"MSE  : {mse:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"MAE  : {mae:.6f}")
        print(f"R2   : {r2:.6f}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    path_data = "smartphone_preprocessing/smartphone_preprocessed.csv"
    target_column = "Price"
    tune_and_log(data_path=path_data, target=target_column)
