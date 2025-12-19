import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from mlflow.models.signature import infer_signature
import warnings

warnings.filterwarnings('ignore')

class MotorPriceTrainer:
    def __init__(self):
        if os.getenv("GITHUB_ACTIONS"):
            self.tracking_uri = "file:./mlruns"
        else:
            self.tracking_uri = "http://127.0.0.1:5000"
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("Motorcycle-Price-Analysis")

    def prepare_data(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_path, 'motor_preprocessed.csv')
        
        df = pd.read_csv(csv_path)
        X = df.drop('selling_price', axis=1)
        y = df['selling_price']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def search_best_params(self, X_train, y_train):
        grid_config = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_leaf': [1, 2]
        }
        regressor = RandomForestRegressor(random_state=42)
        optimizer = GridSearchCV(regressor, grid_config, cv=3, scoring='r2', n_jobs=-1)
        optimizer.fit(X_train, y_train)
        return optimizer.best_estimator_, optimizer.best_params_

    def log_performance_artifacts(self, y_actual, y_predicted):
        errors = y_actual - y_predicted
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, color='purple')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error Amount')
        plot_name = "residual_analysis.png"
        plt.savefig(plot_name)
        mlflow.log_artifact(plot_name)
        plt.close()
        os.remove(plot_name)

    def execute(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        with mlflow.start_run(run_name="Optimized_RF_Model"):
            model, params = self.search_best_params(X_train, y_train)
            mlflow.log_params(params)
            
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            mlflow.log_metrics({"MAE": mae, "R2": r2, "RMSE": rmse})
            
            self.log_performance_artifacts(y_test, preds)
            
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "bike_price_model", signature=signature)
            
            print(f"Training Selesai. R2 Score: {r2:.4f}")

if __name__ == "__main__":
    trainer = MotorPriceTrainer()
    trainer.execute()
