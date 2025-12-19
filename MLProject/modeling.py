import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings('ignore')

def load_data():
    path = 'motor_preprocessed.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Jalankan automate script dulu!")
        
    df = pd.read_csv(path)
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    
    return metrics, y_pred

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Bike-Price-Optimization")
    
    mlflow.sklearn.autolog()
    
    X, y = load_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        model = train_model(X_train, y_train)
        
        metrics, y_pred = evaluate_model(model, X_val, y_val)
        
        print("Model Training Selesai (Autolog)")
        print(f"R2 Score: {metrics['r2_score']:.4f}")

if __name__ == "__main__":
    main()