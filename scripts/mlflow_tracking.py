import mlflow
import joblib
import os
import yaml
from utils import load_config


def mlflow_run(experiment_name, parameters, metrics_path, model_path, data_paths):
    config = load_config('../config/config.yaml')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(parameters)
        
        # Log metrics
        with open(metrics_path, 'r') as f:
            metrics = yaml.safe_load(f)
        
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model_path, "model")
        
        # Log artifacts (data and results)
        for key, path in data_paths.items():
            mlflow.log_artifact(path)

def main():
    current_path = os.getcwd()
    config_path = os.path.join(current_path, 'config/config.yaml')
    config = load_config(config_path)
    experiment_name = config['mlflow']['experiment_name']
    parameters = config['model']['parameters']
    paths = config['paths']
    evaluation_output_path = paths['evaluation_output_path']
    model_output_path = paths['model_output_path']
    X_train_path = os.path.join(current_path, paths['data_output_path'], 'X_train.csv')
    X_val_path = os.path.join(current_path, paths['data_output_path'], 'X_val.csv')
    y_train_path = os.path.join(current_path, paths['data_output_path'], 'y_train.csv')
    y_val_path = os.path.join(current_path, paths['data_output_path'], 'y_val.csv')

    mlflow_run(
        experiment_name,
        parameters, 
        evaluation_output_path,
        model_output_path,
        data_paths={
            'X_train_path': X_train_path,
            'X_val_path': X_val_path,
            'y_train_path': y_train_path,
            'y_val_path': y_val_path
        }
    )

if __name__ == "__main__":
    main()
