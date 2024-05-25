import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import yaml
from utils import load_config

class ModelEvaluator:
    def __init__(self, config_path='../config/config.yaml'):
        self.config = load_config(config_path)
        self.path = self.config['paths']
        current_path = os.getcwd()
        self.model_output_path = os.path.join(current_path, self.path['model_output_path'])
        self.evaluation_output_path = os.path.join(current_path, self.path['evaluation_output_path'])

    def evaluate_model(self, model, X_val_path, y_val_path):
        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path)
        y_val = np.ravel(y_val)
        
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),  # Convert to list for JSON/YAML compatibility
            'recall': float(recall),
            'f1_score': float(f1),
        }

    def run(self, X_val_path, y_val_path):
        model = joblib.load(self.model_output_path)
        metrics = self.evaluate_model(model, X_val_path, y_val_path)
        
        os.makedirs(os.path.dirname(self.evaluation_output_path), exist_ok=True)
        with open(self.evaluation_output_path, 'w') as f:
            yaml.dump(metrics, f)
        
        return self.evaluation_output_path

def main():
    current_path = os.getcwd()
    config_path = os.path.join(current_path, 'config/config.yaml')
    config = load_config(config_path)
    paths = config['paths']
    evaluator = ModelEvaluator(config_path)
    X_val_path = os.path.join(current_path, paths['data_output_path'], 'X_val.csv')
    y_val_path = os.path.join(current_path, paths['data_output_path'], 'y_val.csv')
    evaluation_output_path = evaluator.run(X_val_path, y_val_path)

if __name__ == "__main__":
    main()