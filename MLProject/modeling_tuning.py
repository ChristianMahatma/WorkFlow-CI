import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import estimator_html_repr
from sklearn.model_selection import train_test_split

repo_owner = "ChristianMahatma"
repo_name = "Eksperimen_SML_ChristianMahatmaBimaAlpindo"

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Smartphone_Price_Tuning_Christian")

def load_data(path, target):
    try:
        df = pd.read_csv(path)
        if target not in df.columns:
            raise ValueError(f"Kolom target '{target}' tidak ditemukan!")
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file CSV: {e}")

def tune_and_log(data_path, target, n_iter=6, cv=3):
    X, y = load_data(data_path, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1)
    
    print("Sedang tuning dan melatih model...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name="Advance_Modeling_Christian"):
        # 1. Log Params & Metrics
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model", 
            registered_model_name="SmartphonePricePredictor_Christian"
            
            )

        metrics_dict = {"mae": mae, "mse": mse, "r2_score": r2}
        with open("metric_info.json", "w", encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        with open("estimator.html", "w", encoding='utf-8') as f:
            f.write(estimator_html_repr(best_model))
        mlflow.log_artifact("estimator.html")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, preds, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Price')
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Feature Importance - Top 10")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

    print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    path_data = "smartphone_preprocessing/smartphone_preprocessed.csv"
    target_column = "Price"
    tune_and_log(data_path=path_data, target=target_column)