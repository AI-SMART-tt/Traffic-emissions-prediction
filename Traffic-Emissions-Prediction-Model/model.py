# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2024/3/27 12:50
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')


def data_preprocess(file_path, test_size=0.2, random_state=42):
    """
    Data preprocessing and split into training and test sets
    """
    df = pd.read_csv(file_path, encoding='gbk')

    features = [
        'Speed stddev',
        'Congestion level',
        'Speed',
        'Nonlocal vehicles density',
        'Population density',
        'PGDP',
        'Land-use mixture',
        'Road intersection density',
        'Distance to city center',
        'Industrial land density',
        'Elevated roads and expressway density',
        'Primary industry',
        'Tertiary industry',
        'Elevated bridges and expressway intersection density',
        'Internal road density',
        'Secondary industry',
        'Commercial land density',
        'Suburban arterial road density',
        'Public administration and service land density',
        'Pedestrian path density',
        'Suburban and rural road density',
        'Seaport/dock facility density'
    ]

    X = df[features]
    y = df['avg emission NOx'].values

    # Normalization
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_Y.fit_transform(y.reshape(-1, 1)).flatten()

    hour = df['Hour']
    day_period = df['DayPeriod']
    district = df['district']
    zone = df['Zone']
    pru = df['PRU']

    X_train, X_test, y_train, y_test, hour_train, hour_test, day_period_train, day_period_test, \
    district_train, district_test, zone_train, zone_test, pru_train, pru_test = train_test_split(
        X, y, hour, day_period, district, zone, pru, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, hour_train, hour_test, day_period_train, day_period_test, \
           district_train, district_test, zone_train, zone_test, pru_train, pru_test, scaler_X, scaler_Y, features


def select_model(model_name, **params):
    """
    Model selection function
    """
    if model_name == 'SVR':
        return SVR(
            C=params.get('C', 1.0),
            epsilon=params.get('epsilon', 0.1),
            kernel=params.get('kernel', 'rbf'),
            degree=params.get('degree', 3),
            gamma=params.get('gamma', 'scale'),
            coef0=params.get('coef0', 0.0),
            tol=params.get('tol', 1e-3),
            cache_size=params.get('cache_size', 200),
            verbose=params.get('verbose', False),
            max_iter=params.get('max_iter', -1)
        )
    elif model_name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', 42)
        )
    elif model_name == 'GBDT':
        return GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42)
        )
    elif model_name == 'AdaBoost':
        return AdaBoostRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42)
        )
    elif model_name == 'XGBoost':
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42)
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Model training function
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate model performance on validation set
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    explained_variance = explained_variance_score(y_val, y_val_pred)

    mask = y_val > 0.1
    mape = np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
    mpe = np.mean((y_val[mask] - y_val_pred[mask]) / y_val[mask]) * 100

    r2 = r2_score(y_val, y_val_pred)

    # Calculate adjusted R-squared
    n = len(y_val)
    p = X_val.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return train_time, mae, rmse, mape, mpe, r2, adjusted_r2, explained_variance


def test_model(model, X_test, y_test, scaler_Y):
    """
    Evaluate model performance on test set
    """
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    # Transform predictions back to original scale
    y_test_pred_original = scaler_Y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
    test_rmse = mean_squared_error(y_test_original, y_test_pred_original, squared=False)
    explained_variance = explained_variance_score(y_test_original, y_test_pred_original)

    mask = y_test_original > 0.1
    test_mape = np.mean(np.abs((y_test_original[mask] - y_test_pred_original[mask]) / y_test_original[mask])) * 100
    test_mpe = np.mean((y_test_original[mask] - y_test_pred_original[mask]) / y_test_original[mask]) * 100

    test_r2 = r2_score(y_test_original, y_test_pred_original)

    # Calculate adjusted R-squared
    n = len(y_test_original)
    p = X_test.shape[1]
    test_adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

    return test_time, test_mae, test_rmse, test_mape, test_mpe, test_r2, test_adjusted_r2, explained_variance, y_test_original, y_test_pred_original


def evaluate_by_group(y_test, y_pred, group_test, group_name, X_test):
    """
    Evaluate model performance by specified group
    """
    results = []
    for group in sorted(group_test.unique()):
        mask = group_test == group
        y_test_group = y_test[mask]
        y_pred_group = y_pred[mask]

        mae = mean_absolute_error(y_test_group, y_pred_group)
        rmse = mean_squared_error(y_test_group, y_pred_group, squared=False)

        mask = y_test_group > 0.1
        mape = np.mean(np.abs((y_test_group[mask] - y_pred_group[mask]) / y_test_group[mask])) * 100

        r2 = r2_score(y_test_group, y_pred_group)

        n = len(y_test_group)
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        results.append({
            group_name: group,
            'Test MAE': mae,
            'Test RMSE': rmse,
            'Test MAPE': mape,
            'Test R²': r2,
            'Test Adjusted R²': adjusted_r2
        })

    return pd.DataFrame(results)


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Values"):
    """
    Plot actual values against predicted values with reference line
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='green', alpha=0.6, label='Predicted')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def k_fold_cross_validation(model_name, X, y, k, feature_names, **params):
    """
    Perform K-fold cross-validation for model selection and evaluation
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = select_model(model_name, **params)
        train_time, mae, rmse, mape, mpe, r2, adjusted_r2, explained_variance = train_model(
            model, X_train, y_train, X_val, y_val
        )

        results.append({
            'Fold': fold + 1,
            'Training Time (s)': train_time,
            'Validation MAE': mae,
            'Validation RMSE': rmse,
            'Validation MAPE (%)': mape,
            'Validation MPE (%)': mpe,
            'Validation R²': r2,
            'Validation Adjusted R²': adjusted_r2,
            'Validation Explained Variance': explained_variance
        })

    results_df = pd.DataFrame(results)
    return results_df


def main(file_path, model_name, output_file, test_size=0.2, k=5, random_state=42, **params):
    """
    Main function for model training, evaluation and visualization
    """
    # Data preprocessing
    start_time = time.time()
    X_train, X_test, y_train, y_test, hour_train, hour_test, day_period_train, day_period_test, \
    district_train, district_test, zone_train, zone_test, pru_train, pru_test, scaler_X, scaler_Y, feature_names = data_preprocess(
        file_path, test_size=test_size, random_state=random_state
    )
    data_preprocess_time = time.time() - start_time

    # K-fold cross-validation
    cv_results_df = k_fold_cross_validation(model_name, X_train, y_train, k, feature_names, **params)

    # Model training
    start_time = time.time()
    model = select_model(model_name, **params)
    model.fit(X_train, y_train)
    model_train_time = time.time() - start_time

    # Model testing
    test_time, test_mae, test_rmse, test_mape, test_mpe, test_r2, test_adjusted_r2, \
    test_explained_variance, y_test_original, y_test_pred_original = test_model(model, X_test, y_test, scaler_Y)

    # Collect results
    results_dict = {
        'Model': model_name,
        'Data Preprocess Time (s)': data_preprocess_time,
        'Model Training Time (s)': model_train_time,
        'Model Evaluation Time (s)': test_time,
        'Total Time (s)': data_preprocess_time + model_train_time + test_time,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Test MAPE (%)': test_mape,
        'Test MPE (%)': test_mpe,
        'Test R²': test_r2,
        'Test Adjusted R²': test_adjusted_r2,
        'Test Explained Variance': test_explained_variance
    }

    print("Results:")
    for key, value in results_dict.items():
        print(f"{key}: {value}")

    # Group evaluation results
    hour_results_df = evaluate_by_group(y_test_original, y_test_pred_original, hour_test, 'Hour', X_test)
    day_period_results_df = evaluate_by_group(y_test_original, y_test_pred_original, day_period_test, 'DayPeriod',
                                              X_test)
    district_results_df = evaluate_by_group(y_test_original, y_test_pred_original, district_test, 'district', X_test)
    zone_results_df = evaluate_by_group(y_test_original, y_test_pred_original, zone_test, 'Zone', X_test)
    pru_results_df = evaluate_by_group(y_test_original, y_test_pred_original, pru_test, 'PRU', X_test)

    # Print group evaluation results
    print("\nHourly Test Results:\n", hour_results_df)
    print("\nDayPeriod Test Results:\n", day_period_results_df)
    print("\nDistrict Test Results:\n", district_results_df)
    print("\nZone Test Results:\n", zone_results_df)
    print("\nPRU Test Results:\n", pru_results_df)

    # Visualize test results by hour (MAE and RMSE)
    hour_results_df.set_index('Hour')[['Test MAE', 'Test RMSE']].plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance by Hour - Error Metrics')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Visualize test results by hour (R²)
    hour_results_df.set_index('Hour')[['Test R²']].plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance by Hour - R² Score')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Plot actual vs predicted values
    plot_actual_vs_predicted(y_test_original, y_test_pred_original, f"{model_name} - Actual vs Predicted Values")

    # Return results and model
    return results_dict, model, cv_results_df, hour_results_df, day_period_results_df, district_results_df, zone_results_df, pru_results_df, y_test_original, y_test_pred_original