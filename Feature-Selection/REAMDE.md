# Pollutant Emission Analysis Pipeline

## Overview
This project implements a comprehensive machine learning pipeline for analyzing and predicting pollutant emissions. It provides tools for data processing, feature engineering, multicollinearity handling, and feature selection to build optimal predictive models for various emissions (CO, NOx, PM, VOC).

## Key Features
- Automated data loading and missing value handling
- Feature classification into meaningful categories
- Multicollinearity detection and resolution
- Multiple feature scaling options (Robust, Standard, MinMax)
- Advanced feature selection using LightGBM and SHAP values
- Cross-validation for reliable feature importance evaluation

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- lightgbm
- shap
- statsmodels

## Installation
```bash
pip install pandas numpy scikit-learn lightgbm shap statsmodels
```

## Usage
1. Prepare your emissions data CSV file
2. Configure parameters in the params dictionary
3. Run the script to process data and select optimal features

```python
# Example usage
params = {
    'csv_path': 'your_emission_data.csv',
    'encoding': 'utf-8',
    'missing_value_strategy': 'interpolate',
    'target_name': 'avg emission NOx',
    # Other parameters...
}

# Run the pipeline
processor = DataProcessor(params['csv_path'], params['encoding'])
# Continue with the remaining steps as shown in the main script
```

## Pipeline Components
1. **DataProcessor**: Handles loading and preprocessing of emission data
2. **FeatureClassifier**: Organizes features into meaningful categories
3. **MulticollinearitySolver**: Detects and resolves feature correlations
4. **FeatureScaler**: Normalizes features for better model performance
5. **FeatureSelectionUsingLightGBM**: Selects optimal feature subset using advanced ML techniques

## Output
The pipeline generates:
- Log of processing steps and decisions
- CSV file with best feature weights
- Selected feature set optimized for the target pollutant

## Customization
You can customize the pipeline by:
- Defining your own feature categories
- Specifying features to retain or exclude
- Adjusting importance weighting between native feature importance and SHAP values
- Changing the scaling method used