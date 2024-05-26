import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from utils import load_config

class Preprocessor:
    def __init__(self, config_path='../config/config.yaml'):
        self.config = load_config(config_path)
        self.path = self.config['paths']
        current_path = os.getcwd()
        self.output_path = os.path.join(current_path, self.path['data_output_path'])

    def preprocess_data(self, df):
        df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
        df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())
        df.loc[:, 'Embarked'] = df['Embarked'].fillna('S')
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 1
        df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
        return df

    def run(self, train_df):
        train_df = self.preprocess_data(train_df)
        X = train_df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        y = train_df['Survived']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save processed data
        os.makedirs(self.output_path, exist_ok=True)
        X_train_path = os.path.join(self.output_path, 'X_train.csv')
        X_val_path = os.path.join(self.output_path, 'X_val.csv')
        y_train_path = os.path.join(self.output_path, 'y_train.csv')
        y_val_path = os.path.join(self.output_path, 'y_val.csv')
        
        X_train.to_csv(X_train_path, index=False)
        X_val.to_csv(X_val_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_val.to_csv(y_val_path, index=False)
        
        return X_train_path, X_val_path, y_train_path, y_val_path

def main():
    current_path = os.getcwd()
    print(current_path)
    config_path = os.path.join(current_path, 'config/config.yaml')
    config = load_config(config_path)
    train_df = pd.read_csv(os.path.join(current_path, config["paths"]["data_input_path"]))
    preprocessor = Preprocessor(config_path)
    X_train_path, X_val_path, y_train_path, y_val_path = preprocessor.run(train_df)

if __name__ == "__main__":
    main()