# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2024/3/27 12:41
# ------------------------------------------------------------------------------
import pandas as pd
import shap
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap
from IPython.display import display, Markdown
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import KFold

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


class DataProcessor:
    def __init__(self, csv_path, encoding='utf-8'):
        self.csv_path = csv_path
        self.encoding = encoding
        self.data = self.load_data()

    def load_data(self):
        try:
            data = pd.read_csv(self.csv_path, encoding=self.encoding)
            logging.info("Data loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def detect_missing_values(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            missing_values_percent = (self.data.isnull().sum() / len(self.data)) * 100
            logging.info("Missing values statistics:")
            logging.info("\n" + missing_values.to_string())
            logging.info("Missing values percentage:")
            logging.info("\n" + missing_values_percent.to_string())
        else:
            logging.error("Data not loaded, cannot detect missing values.")

    def fill_missing_values_by_column(self, strategy='interpolate'):
        if self.data is not None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                try:
                    if strategy == 'interpolate':
                        self.data[col] = self.data[col].interpolate(method='linear')
                    else:
                        self.data[col] = self.data[col].fillna(method=strategy)
                except ValueError:
                    logging.warning(f"Cannot fill missing values for column '{col}'. Skipping.")
            logging.info("Missing values filled.")
            return self.data
        else:
            logging.error("Data not loaded, cannot fill missing values.")
            return None


class FeatureClassifier:
    def __init__(self, manual_classification=True):
        self.manual_classification = manual_classification

    def classify_features(self, feature_names, manual_classification_dict=None):
        if self.manual_classification:
            if manual_classification_dict is None:
                raise ValueError("Please provide a manual classification dictionary.")
            classifications = {}
            for class_name, features in manual_classification_dict.items():
                for feature in features:
                    classifications[feature] = class_name
            return classifications
        else:
            prompt = f"""
            Our goal is to classify pollutant emissions. Here is a list of feature names: {', '.join(feature_names)}.
            Please classify them into up to M categories based on their meanings, and provide category names.
            Also, identify which feature(s) should be considered as the target variable (y).
            Return the results in the format:
            "Category1: [Feature1, Feature2, ...]
            Category2: [Feature1, Feature2, ...]
            Target: [Target1, Target2, ...]"
            """
            response = model.generate_content(prompt)
            classification_text = response.text
            display(to_markdown(classification_text))

            classifications = {}
            for line in classification_text.splitlines():
                if line.strip() and ':' in line:
                    category, features_str = line.split(':', 1)
                    category = category.strip()
                    features = [feature.strip() for feature in features_str.strip().strip('[]').split(',')]
                    for feature in features:
                        if feature in feature_names:
                            classifications[feature] = category
            return classifications

    @staticmethod
    def load_features_by_class(data, feature_classification):
        class_features = {}
        for feature, class_name in feature_classification.items():
            if class_name not in class_features:
                class_features[class_name] = []
            class_features[class_name].append(feature)

        categorized_data = {}
        for class_name, features in class_features.items():
            categorized_data[class_name] = data[features]

        return categorized_data

    @staticmethod
    def initialize_feature_weights(feature_names, selected_features, excluded_features=None):
        if excluded_features is None:
            excluded_features = []

        feature_weights = {}
        for feature in feature_names:
            if feature in selected_features:
                feature_weights[feature] = 1
            elif feature in excluded_features:
                feature_weights[feature] = -1
            else:
                feature_weights[feature] = 0

        return feature_weights


class MulticollinearitySolver:
    def __init__(self, data, selected_features, feature_weights, handle_multicollinearity=True):
        self.data = data
        self.selected_features = selected_features
        self.feature_weights = feature_weights
        self.handle_multicollinearity = handle_multicollinearity
        self.iteration = 0

    def get_combined_dataframe(self):
        return pd.concat(self.data.values(), axis=1)

    def calculate_group_correlation(self):
        logging.info("Calculating within-group correlation coefficients")
        for class_name, features in self.data.items():
            corr_matrix = features.corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > 0.95:
                        if feature1 in self.selected_features and feature2 in self.selected_features:
                            logging.info(
                                f"Features {feature1} and {feature2} are both in selected_features, correlation coefficient is {corr_value}, both retained")
                        elif feature1 in self.selected_features:
                            self.feature_weights[feature2] = -1
                            logging.info(
                                f"Features {feature1} (selected) and {feature2} correlation coefficient is {corr_value}, {feature2} removed")
                        elif feature2 in self.selected_features:
                            self.feature_weights[feature1] = -1
                            logging.info(
                                f"Features {feature2} (selected) and {feature1} correlation coefficient is {corr_value}, {feature1} removed")
            logging.info(f"Updated weights for category {class_name} after calculating within-group correlation:")
            logging.info({feature: self.feature_weights[feature] for feature in features.columns})

    def calculate_vif(self, threshold=10):
        combined_df = self.get_combined_dataframe()
        features = [feature for feature, weight in self.feature_weights.items() if weight >= 0]
        while True:
            self.iteration += 1
            logging.info(f"\nIteration {self.iteration}:")
            vif_data = pd.DataFrame()
            vif_data['Feature'] = features
            vif_data['VIF'] = [variance_inflation_factor(combined_df[features].values, i) for i in range(len(features))]

            non_selected_features_vif = vif_data[~vif_data['Feature'].isin(self.selected_features)]
            if non_selected_features_vif.empty:
                break

            max_vif = non_selected_features_vif['VIF'].max()
            max_vif_feature = non_selected_features_vif.loc[non_selected_features_vif['VIF'].idxmax(), 'Feature']

            if max_vif > threshold:
                self.feature_weights[max_vif_feature] = -1
                features.remove(max_vif_feature)
                logging.info(f"Removed feature: {max_vif_feature}, VIF value: {max_vif}")
            else:
                break

            logging.info("Updated weights:")
            logging.info({feature: weight for feature, weight in self.feature_weights.items() if weight != -1})

        logging.info("\nFinal VIF values for retained features:")
        remaining_features = [feature for feature, weight in self.feature_weights.items() if weight >= 0]
        vif_data = pd.DataFrame()
        vif_data['Feature'] = remaining_features
        vif_data['VIF'] = [variance_inflation_factor(combined_df[remaining_features].values, i) for i in
                           range(len(remaining_features))]
        logging.info(vif_data)

    def solve(self):
        if self.handle_multicollinearity:
            logging.info("Handling multicollinearity...")
            self.calculate_group_correlation()
            self.calculate_vif(threshold=10)
        else:
            logging.info("Skipping multicollinearity handling.")


class FeatureScaler:
    def __init__(self, data, feature_weights, target_data):
        self.data = data
        self.feature_weights = feature_weights
        self.target_data = target_data
        self.scaler_X = None
        self.scaler_Y = None
        self.original_features = None
        self.features = None

    def consolidate_features(self):
        self.features = [feature for feature, weight in self.feature_weights.items() if weight != -1]
        consolidated_data = self.data[self.features]
        logging.info("Consolidated features:")
        logging.info(consolidated_data)
        logging.info("Feature consolidation completed.")
        return consolidated_data

    def scale_and_split_data(self, method='RobustScaler', test_size=0.2, random_state=42):
        if method == 'RobustScaler':
            self.scaler_X = RobustScaler()
            self.scaler_Y = RobustScaler()
        elif method == 'StandardScaler':
            self.scaler_X = StandardScaler()
            self.scaler_Y = StandardScaler()
        elif method == 'MinMaxScaler':
            self.scaler_X = MinMaxScaler()
            self.scaler_Y = MinMaxScaler()
        else:
            raise ValueError(
                "Invalid scaling method. Please choose 'RobustScaler', 'StandardScaler', or 'MinMaxScaler'.")

        self.original_features = self.data.copy()

        consolidated_data = self.consolidate_features()
        scaled_features = self.scaler_X.fit_transform(consolidated_data)
        scaled_target = self.scaler_Y.fit_transform(self.target_data)

        scaled_features_df = pd.DataFrame(scaled_features, columns=consolidated_data.columns)
        scaled_target_df = pd.DataFrame(scaled_target, columns=self.target_data.columns)

        logging.info("Features after standardization:")
        logging.info(scaled_features_df)
        logging.info("Target variables after standardization:")
        logging.info(scaled_target_df)
        logging.info("Standardization completed.")

        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features_df, scaled_target_df, test_size=test_size, random_state=random_state
        )

        logging.info(f"Training set feature shape: {X_train.shape}, test set feature shape: {X_test.shape}")
        logging.info(f"Training set target shape: {y_train.shape}, test set target shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test, self.scaler_X, self.scaler_Y, self.features


class FeatureSelectionUsingLightGBM:
    def __init__(self, X_train, X_test, y_train, y_test, feature_weights, importance_weight=0.5, num_folds=5,
                 random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_weights = feature_weights
        self.importance_weight = importance_weight
        self.num_folds = num_folds
        self.random_state = random_state
        self.iteration = 0
        self.best_iteration = None
        self.best_error = float('inf')
        self.best_feature_weights = None
        self.best_features = None
        self.removed_features = []

    def normalize_importance(self, importance):
        max_val = max(importance)
        min_val = min(importance)
        return [(i - min_val) / (max_val - min_val) for i in importance]

    def calculate_shap_values(self, model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_importance = np.abs(shap_values).mean(axis=0)
        return shap_importance

    def train_lightgbm(self, features):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        importance_values = np.zeros(len(features))
        shap_values = np.zeros(len(features))
        errors = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_index][features], self.X_train.iloc[val_index][features]
            y_tr, y_val = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            model = lgb.LGBMRegressor(random_state=self.random_state)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mse')

            importance_values += model.feature_importances_
            shap_importance = self.calculate_shap_values(model, X_val)
            shap_values += shap_importance

            y_pred = model.predict(X_val)
            errors.append(mean_squared_error(y_val, y_pred))

        avg_error = np.mean(errors)

        normalized_importance = self.normalize_importance(importance_values)
        normalized_shap_values = self.normalize_importance(shap_values)

        combined_importance = [(self.importance_weight * imp + (1 - self.importance_weight) * shap)
                               for imp, shap in zip(normalized_importance, normalized_shap_values)]

        return combined_importance, avg_error

    def update_feature_weights(self, features, importance):
        for i, feature in enumerate(features):
            if self.feature_weights[feature] == 0:
                self.feature_weights[feature] = importance[i]

        sorted_weights = {k: v for k, v in sorted(self.feature_weights.items(), key=lambda item: item[1], reverse=True)}
        logging.info(f"Weight ranking after iteration {self.iteration + 1}: {sorted_weights}")
        return sorted_weights

    def remove_least_important_feature(self):
        min_weight_feature = min(self.feature_weights,
                                 key=lambda k: self.feature_weights[k] if self.feature_weights[k] > 0 else float('inf'))
        self.feature_weights[min_weight_feature] = -1
        self.removed_features.append(min_weight_feature)
        logging.info(f"Removed feature: {min_weight_feature}")

    def run(self):
        while len([w for w in self.feature_weights.values() if w > 0]) > 1:
            self.iteration += 1
            current_features = [f for f, w in self.feature_weights.items() if w >= 0]
            normalized_importance, avg_error = self.train_lightgbm(current_features)
            self.feature_weights = self.update_feature_weights(current_features, normalized_importance)

            if avg_error < self.best_error:
                self.best_error = avg_error
                self.best_iteration = self.iteration
                self.best_feature_weights = self.feature_weights.copy()
                self.best_features = current_features

            self.remove_least_important_feature()
            logging.info(f"Error value after iteration {self.iteration}: {avg_error}")

        logging.info(f"\nIteration with minimal error: {self.best_iteration}")
        logging.info(f"Previously removed features: {self.removed_features[:self.best_iteration - 1]}")
        logging.info(f"Features used for training: {self.best_features}")
        logging.info(f"Feature weights: {self.best_feature_weights}")
        self.output_best_feature_weights_to_csv('best_feature_weights.csv')
        return self.removed_features[:self.best_iteration - 1], self.best_feature_weights

    def output_best_feature_weights_to_csv(self, filename):
        df = pd.DataFrame({
            'Feature': list(self.best_feature_weights.keys()),
            'Weight': list(self.best_feature_weights.values())
        })
        df.to_csv(filename, index=False)
        logging.info(f"Best feature weights saved to {filename}")


if __name__ == "__main__":
    params = {
        'csv_path': 'all_new_features_1h_NEW.csv',
        'encoding': 'utf-8',
        'missing_value_strategy': 'interpolate',
        'test_size': 0.2,
        'random_state': 42,
        'target_name': 'avg emission NOx',

        'target_variables': ['vehicle emission CO', 'vehicle emission NOx', 'vehicle emission PM',
                             'vehicle emission VOC',
                             'avg emission CO', 'avg emission NOx', 'avg emission PM', 'avg emission VOC'],
        'selected_features': ['Speed stddev', 'PGDP', 'Distance to city center', 'Speed',
                              'Congestion level', 'Land-use mixture', 'Population density',
                              'Road intersection density'],
        'excluded_features': ['Avg-speed change rate', 'Nonlocal vehicles density', 'Local vehicles density',
                              'China3 density',
                              'China4 density', 'China5 density', 'China6 density', 'Vehicle1 density',
                              'Vehicle2 density',
                              'Vehicle3 density', 'Vehicle4 density'],
        'extra_columns': ['Time', 'Street', 'area', 'district', 'districtID', 'Hour', 'Long', 'Lat', 'DayPeriod',
                          'Inner Ring', 'Outer Ring', 'Middle Ring', 'Zone', 'PRU'],

        'manual_classification': True,
        'handle_multicollinearity': False,
        'importance_weight': 0,
        'scaling_method': 'MinMaxScaler',

        'manual_classification_dict': {
            'Class1_Built_environment': [
                'Primary industry', 'Secondary industry', 'Tertiary industry',
                'Industrial land density', 'Public administration and service land density',
                'Transportation land density', 'Residential land density', 'Commercial land density',
                'Land-use mixture', 'Suburban arterial road density', 'Urban collector road density',
                'Urban arterial road density', 'Elevated roads and expressway density',
                'Suburban and rural road density', 'Internal road density',
                'Pedestrian path density', 'Bicycle lane density', 'Road density',
                'Elevated bridges and expressway intersection density',
                'Urban arterial road intersection density', 'Suburban arterial road intersection density',
                'Road intersection density',
                'Distance to city center', 'Time to city center', 'Parking facility density',
                'Bus stop density', 'Metro station density',
                'Airport-related infrastructure density', 'Seaport/dock facility density',
                'Train station density', 'Ferry terminal density'
            ],
            'Class2_Vehicle_driving_characteristics': [
                'Avg-speed change rate', 'Speed range', 'Speed stddev'
            ],
            'Class3_Demographic_characteristics': [
                'PGDP', 'Population density'
            ],
            'Class4_Traffic_state': [
                'Congestion level', 'Speed'
            ],
            'Class5_Vehicle_type': [
                'China3 density', 'China4 density', 'China5 density', 'China6 density',
                'Vehicle1 density', 'Vehicle2 density', 'Vehicle3 density', 'Vehicle4 density',
                'Nonlocal vehicles density', 'Local vehicles density'
            ]
        }
    }

    # Read and process data
    processor = DataProcessor(params['csv_path'], params['encoding'])
    processor.detect_missing_values()
    data = processor.fill_missing_values_by_column(strategy=params['missing_value_strategy'])

    # Extract extra spatiotemporal information fields
    extra_columns = params['extra_columns']
    extra_data = data[extra_columns]
    data = data.drop(columns=extra_columns)

    # Separate target variables
    target_data = data[params['target_variables']]
    data = data.drop(columns=params['target_variables'])
    feature_names = list(data.columns)

    # Feature classification
    classifier = FeatureClassifier(manual_classification=params['manual_classification'])
    feature_classification = classifier.classify_features(feature_names, params['manual_classification_dict'])
    categorized_data = FeatureClassifier.load_features_by_class(data, feature_classification)
    feature_weights = FeatureClassifier.initialize_feature_weights(feature_names, params['selected_features'],
                                                                   params['excluded_features'])

    # Handle multicollinearity
    solver = MulticollinearitySolver(categorized_data, params['selected_features'], feature_weights,
                                     handle_multicollinearity=params['handle_multicollinearity'])
    solver.solve()

    logging.info("\nFinal feature weights:")
    for feature, weight in feature_weights.items():
        logging.info(f"{feature}: {weight}")

    logging.info("\nTarget variables:")
    logging.info(target_data)

    # Count weight distribution
    weights_count = {1: 0, 0: 0, -1: 0}
    for weight in feature_weights.values():
        if weight in weights_count:
            weights_count[weight] += 1
    logging.info(
        f"Features with weight 1: {weights_count[1]}, with weight 0: {weights_count[0]}, with weight -1: {weights_count[-1]}")

    # Instantiate feature scaler and consolidate features
    scaler = FeatureScaler(data, feature_weights, target_data)
    consolidated_data = scaler.consolidate_features()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test, scaler_X, scaler_Y, features_name = scaler.scale_and_split_data(
        test_size=params['test_size'], random_state=params['random_state'], method=params['scaling_method'])

    # Select one target variable
    target_name = params['target_name']
    y_train = y_train[[target_name]]
    y_test = y_test[[target_name]]

    #################### Data processing completed ####################

    # Train model and select optimal feature combination
    feature_selector = FeatureSelectionUsingLightGBM(X_train, X_test, y_train, y_test, feature_weights,
                                                     importance_weight=params['importance_weight'])
    removed_features, best_feature_weights = feature_selector.run()
    # Features to keep
    features_to_keep = [feature for feature, weight in best_feature_weights.items() if weight != -1]