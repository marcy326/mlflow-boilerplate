import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from utils import load_config

class ModelTrainer:
    def __init__(self, config_path='../config/config.yaml'):
        self.config = load_config(config_path)
        self.path = self.config['paths']
        current_path = os.getcwd()
        self.model_output_path = os.path.join(current_path, self.path['model_output_path'])

    def train_model(self, X_train_path, y_train_path, parameters):
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train = np.ravel(y_train)
        
        model = RandomForestClassifier(**parameters)
        model.fit(X_train, y_train)
        
        return model

    def run(self, X_train_path, y_train_path, parameters):
        model = self.train_model(X_train_path, y_train_path, parameters)
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(model, self.model_output_path)
        return self.model_output_path

def main():
    current_path = os.getcwd()
    config_path = os.path.join(current_path, 'config/config.yaml')
    config = load_config(config_path)
    paths = config['paths']
    X_train_path = os.path.join(current_path, paths['data_output_path'], 'X_train.csv')
    y_train_path = os.path.join(current_path, paths['data_output_path'], 'y_train.csv')
    parameters = config['model']['parameters']
    trainer = ModelTrainer(config_path)
    model_output_path = trainer.run(X_train_path, y_train_path, parameters)

if __name__ == "__main__":
    main()