
# Telco Customer Churn Prediction Project
# Author: Afeef
# Module: CM2604 Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_data(filepath='telco_churn.csv'):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

def preprocess_data(df):
    data = df.copy()
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['MonthlyCharges'] * data['tenure'], inplace=True)
    
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
        
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data = pd.get_dummies(data, drop_first=True)
    
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    print("Training Neural Network...")
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    return nn, dt

if __name__ == "__main__":
    print("Starting Pipeline...")
    # Note: This script assumes 'telco_churn.csv' is in the same folder
    df = load_data('telco_churn.csv')
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        nn_model, dt_model = train_models(X_train, y_train)
        print("Pipeline Complete. Models Trained.")
