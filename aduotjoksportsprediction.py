# -*- coding: utf-8 -*-
"""AduotJokSportsPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CvgGclbgpk_wEEtTVqQcwZ5k_WzS8c8v
"""

# This cell will import the drive
from google.colab import drive

# This cell acts as a connection between Google drive and Colab
drive.mount('/content/drive')

# This cell holds some important imports
import pandas as pd
import pickle
import zipfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from matplotlib import pyplot as plt

"""Import the dataSet"""

data_import1 = pd.read_csv('/content/drive/MyDrive/DataSet/male_players.csv')

data_import1.head(2)

"""#1 Demonstrate the data preparation & feature extraction process"""

# This full will do the preprocessing of the data_import1
def preprocess_data(data_import1):
  # Drop columns with too many missing values
  data_import1.dropna(thresh=len(data_import1) * 0.3, axis=1, inplace=True)
  columns_to_drop = data_import1.filter(regex='age|url|club|id|eur|name|player|face|body|worker|nation|national|league|dob|fifa|work|foot|mentality|goalkeeping').columns

  # Drop the columns
  df = data_import1.drop(columns=columns_to_drop)
  df.drop(['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
        'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb',
        'lcb', 'cb', 'rcb', 'rb', 'gk'], axis=1, inplace=True)
  return df

"""# 2 Create feature subsets that show maximum correlation with the dependent variable"""

#This function will select the best features in the data_import1
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def feature_selection(data):
    Y = data['overall']
    X = data.drop(['overall'], axis=1)
    X_imputed = X.fillna(X.mean())

    # Apply SelectKBest with f_regression to select top 7 features
    selector = SelectKBest(score_func=f_regression, k=7)
    X_new = selector.fit_transform(X_imputed, Y)

    # Get the columns selected by SelectKBest
    selected_columns = X_imputed.columns[selector.get_support()]

    # Convert the selected features to a DataFrame
    X_new_df = pd.DataFrame(X_new, columns=selected_columns)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new_df)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Convert the scaled features back to a DataFrame to retain column names
    X_train_df = pd.DataFrame(X_train, columns=selected_columns)
    X_test_df = pd.DataFrame(X_test, columns=selected_columns)

    return (X_train_df, X_test_df, Y_train, Y_test)

"""# 3 Create and train a suitable machine learning model with cross-validation that can predict a player's rating."""

#This function will train the models that would be used later in the testing
import joblib
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

def train_models(X_train, X_test, Y_train, Y_test):
    # Ensure input data is in DataFrame format
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Check for NaN values in the data
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        print("NaN values detected. Proceeding with models that can handle NaN values natively...")

    # Define models
    models = {
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'XGBRegressor': XGBRegressor()
    }

    # Define parameter grids for each model
    param_grids = {
        'HistGradientBoostingRegressor': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'max_iter': [10, 50]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [10, 50],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'XGBRegressor': {
            'n_estimators': [10, 50],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    # Initialize dictionaries to store best models and their evaluation metrics
    best_models = {}
    best_metrics = {}

    # Perform GridSearchCV for each model
    for model_name, model in models.items():
        print(f"Performing GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)

        # Store the best model and its metrics
        best_models[model_name] = best_model
        best_metrics[model_name] = {'MAE': mae, 'R2': r2}

        print(f"Metrics for {model_name}: MAE: {mae}, R2: {r2}")

    # Convert best_metrics to DataFrame
    metrics_df = pd.DataFrame(best_metrics).T

    # Find the best model with the lowest MAE
    best_model_name = metrics_df['MAE'].idxmin()
    best_model = best_models[best_model_name]
    best_metrics_values = best_metrics[best_model_name]

    print(f"\nBest Model: {best_model_name} with Metrics: {best_metrics_values}")

    # Save the best model of all three models
    joblib.dump(best_model, 'best_model.pkl')

    return

"""# 4 Measure the model's performance and fine-tune it as a process of optimization"""

# This cell below will show the data preprocessing
data = preprocess_data(data_import1)
data.head(2)

#This cell will show the index or heading for the best features in the data
training_data = feature_selection(data)
training_colums = training_data[0].columns
training_data[0].head(2)

trained_model = train_models(training_data[0], training_data[1], training_data[2], training_data[3])

"""Best Model: GradientBoostingRegressor with Metrics: {'MAE': 1.8286172335948403, 'R2': 0.8752759109889612}

# 5 Use the data from another season(players_22) which was not used during the training to test how good is the model

#Importing the testing data
"""

# This cell will import the data from google drive
test_data = pd.read_csv('/content/drive/MyDrive/DataSet/players_22.csv')

#Show the heading
test_data.head(2)

# This cell just keepin the best model saved on 'best_model.pkl')
best_model = joblib.load('best_model.pkl')
best_model

# This combined cell contains the chosen y-test and x values
Y = test_data['overall']
X = test_data[training_colums]
X
X_imputed = X.fillna(X.mean())

# Saving the scaler object here
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Saving the scaler object here
joblib.dump(scaler, 'scaler.pkl')

# Convert the scaled features back to a DataFrame to retain column names
X_new_df = pd.DataFrame(X_scaled, columns=training_colums)

X_new_df.head(2)

# This cell will apply the trained model (best_model) to new, unseen data (X_new_df) and obtain predictions (y_pred).
y_pred = best_model.predict(X_new_df)
y_pred

# This function compare_prediction metrics and print out the best results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compare_predictions(Y_test, y_pred):
    # Create a DataFrame to compare actual and predicted values
    comparison_df = pd.DataFrame({
        'Actual': Y_test,
        'Predicted': y_pred
    })

    # Calculate evaluation metrics
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R2 Score: {r2}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.show()

    return comparison_df

# This cell will call the comparison function above
comparison_df = compare_predictions(Y, y_pred)

"""# 6 Deploy the model on a simple web page using either (Heroku, Streamlite, or Flask) and upload a video that shows how the model performs on the web page/site"""