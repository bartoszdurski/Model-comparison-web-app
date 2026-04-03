# Streamlit Machine Learning Dashboard

## Overview
This repository contains an interactive web application built with Streamlit that allows users to train, tune, and evaluate Machine Learning models on tabular data. It provides an end-to-end pipeline from data imputation to model evaluation, designed for both classification and regression tasks.

## Features
* Interactive Interface: Easy-to-use web dashboard created with Streamlit.
* Multiple Algorithms: Supports popular models including Logistic/Linear Regression, Decision Trees, Random Forests, XGBoost, and CatBoost.
* Missing Data Handling: Includes built-in options for data imputation (e.g., SimpleImputer, IterativeImputer).
* Hyperparameter Tuning: Integrated GridSearchCV for finding the best model parameters.
* Visual Evaluation: Uses Plotly to generate interactive charts such as Feature Importance, Confusion Matrix (for classification), and Actual vs Predicted plots (for regression).

## Project Structure
* app.py: The main Streamlit application script containing the UI and the machine learning logic.
* train.csv: A sample tabular dataset (Titanic dataset) used to demonstrate the classification pipeline.
* catboost_info/: An automatically generated directory containing training logs, metadata, and error metrics produced by the CatBoost algorithm during model training.

## Requirements
To run this project locally, you will need Python installed along with the following primary libraries:
* streamlit
* pandas
* numpy
* scikit-learn
* xgboost
* catboost
* plotly

## Installation and Usage

1. Clone this repository to your local machine.
2. Install the required dependencies. It is recommended to use a virtual environment:
   pip install streamlit pandas numpy scikit-learn xgboost catboost plotly
3. Run the Streamlit application:
   streamlit run app.py
4. Open your web browser and navigate to the local URL provided in the terminal (usually http://localhost:8501) to interact with the dashboard.
