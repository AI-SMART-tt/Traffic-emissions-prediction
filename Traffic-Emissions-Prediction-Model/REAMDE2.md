# Emissions Prediction Model

## Overview
This repository contains a machine learning pipeline for predicting NOx emissions based on various urban and traffic features. The system supports multiple regression models, cross-validation, and comprehensive evaluation metrics with visualizations grouped by temporal and spatial factors.

## Features
- Multiple regression model support (SVR, Random Forest, GBDT, AdaBoost, XGBoost)
- K-fold cross-validation
- Comprehensive evaluation metrics (MAE, RMSE, MAPE, R², Adjusted R²)
- Group-based performance analysis (by hour, day period, district, zone, and PRU)
- Data preprocessing with normalization
- Visualization tools for model performance

## Requirements
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

## Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Usage
```python
from emissions_model import main

# Example usage with Random Forest
results, model, cv_results, hour_results, day_period_results, district_results, zone_results, pru_results, y_test, y_pred = main(
    file_path="your_data.csv",
    model_name="RandomForest",
    output_file="results.csv",
    test_size=0.2,
    k=5,
    random_state=42,
    n_estimators=100,
    max_depth=10
)

# Example usage with XGBoost
results, model, cv_results, hour_results, day_period_results, district_results, zone_results, pru_results, y_test, y_pred = main(
    file_path="your_data.csv",
    model_name="XGBoost",
    output_file="results.csv",
    test_size=0.2,
    k=5,
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
```

## Data Format
The input CSV file should include the following columns:
- Target variable: `avg emission NOx`
- Features: Traffic, urban, and land-use variables (see code for complete list)
- Group variables: `Hour`, `DayPeriod`, `district`, `Zone`, `PRU`

## Key Functions
- `data_preprocess()`: Loads and normalizes data, splits into training and test sets
- `select_model()`: Creates model instances with specified hyperparameters
- `train_model()`: Trains models and evaluates on validation data
- `test_model()`: Evaluates models on test data
- `evaluate_by_group()`: Analyzes model performance across different categorical groups
- `k_fold_cross_validation()`: Performs k-fold CV for robust model evaluation
- `main()`: Orchestrates the entire workflow and returns results

## Output
The pipeline generates:
- Performance metrics dictionary
- Trained model
- Cross-validation results dataframe
- Group-based evaluation dataframes
- Visualizations of model performance by hour
- Actual vs. predicted value plots

## Example Visualization
The code automatically generates bar charts showing model performance metrics (MAE, RMSE, R²) grouped by hour, and scatter plots comparing actual vs. predicted values.