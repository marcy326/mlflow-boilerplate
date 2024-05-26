# MLFlow Boilerplate

This repository contains a machine learning workflow for the Titanic competition on Kaggle. The workflow includes data preprocessing, model training, evaluation, and logging using MLFlow, all orchestrated through GitHub Actions for continuous integration and deployment (CI/CD).

## CI/CD with GitHub Actions
The repository includes a GitHub Actions workflow that automatically runs the preprocessing, training, evaluation, and logging steps on every push or pull request to the main branch. This ensures that your model training pipeline is executed in a consistent environment, regardless of the developer’s local setup.

### Prerequisites
1. MLFlow Tracking Server:  
Set up an MLFlow tracking server to log experiments.

1. GitHub Secrets:  
Set up the following secrets in your GitHub repository settings:
	- MLFLOW_TRACKING_URI: The URI of your MLFlow tracking server.

	- MLFLOW_TRACKING_USERNAME: Your MLFlow tracking server username (if authentication is required).
	
	- MLFLOW_TRACKING_PASSWORD: Your MLFlow tracking server password (if authentication is required).
