import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures
from category_encoders import target_encoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression
from pyod.models.knn import KNN
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeRegressor

# Load and clean dataset
flight = pd.read_csv("Clean_Dataset.csv")
flight.drop("Unnamed: 0", axis=1, inplace=True)

# Split data using stratified sampling based on 'class' for balanced distribution
train_set, test_set = train_test_split(flight, stratify=flight["class"], random_state=42, test_size=0.2)

train_features = train_set.drop("price", axis=1)
train_labels = train_set["price"]

# Alternative split using hash for reproducible splitting (useful for future data additions)
def is_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data[in_test_set]

# Example usage (commented out):
# flight_with_id = flight.reset_index()
# train_set, test_set = split_data_with_id_hash(flight_with_id, 0.2, "index")


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories = [
            ["zero", "one", "two_or_more"],
            ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
        ]
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder(categories=self.categories)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1, random_state=42, whiten=True, svd_solver="randomized")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "stops" in X.columns and "arrival_time" in X.columns:
            X[["stops", "arrival_time"]] = self.ordinal_encoder.fit_transform(X[["stops", "arrival_time"]])
        for col in X.columns:
            if X[col].dtype == "object" and col not in ["stops", "arrival_time"]:
                onehot_encoded = self.onehot_encoder.fit_transform(X[[col]])
                scaled = self.scaler.fit_transform(onehot_encoded)
                pca_transformed = self.pca.fit_transform(scaled)
                pca_columns = [f"pca{num + 1}" for num in range(pca_transformed.shape[1])]
                pca_df = pd.DataFrame(pca_transformed, index=X.index, columns=pca_columns)
                X[col] = pca_df
        return X


class Standardization(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = X.copy()
        self.scaler.fit(X[["duration", "flight", "days_left"]])
        return self

    def transform(self, X):
        X = X.copy()
        X[["duration", "flight", "days_left"]] = self.scaler.transform(X[["duration", "flight", "days_left"]])
        return X


# Function to remove outliers using IQR method
def remove_outliers_with_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data = data[(data >= lower) & (data <= upper)]
    return data


class RemoveOutliersAndAddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_labels):
        self.poly = PolynomialFeatures(interaction_only=True, degree=2)
        self.selector = SelectKBest(k=10, score_func=f_regression)
        self.target_labels = target_labels
        self.isolation_forest = IsolationForest(n_jobs=-1, n_estimators=120, contamination="auto", random_state=42)

    def fit(self, X, y=None):
        X = X.copy()
        self.isolation_forest.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        outliers = self.isolation_forest.predict(X)
        X = X[outliers == 1]
        self.target_labels = self.target_labels[outliers == 1]

        new_features = self.poly.fit_transform(X, self.target_labels)
        selected_features = self.selector.fit_transform(new_features, self.target_labels)
        columns = self.poly.get_feature_names_out()[self.selector.get_support()]
        best_features_df = pd.DataFrame(selected_features, index=X.index, columns=columns)
        best_features_df["target_labels"] = self.target_labels
        valid_idx = remove_outliers_with_iqr(best_features_df["class duration"]).index
        best_features_df = best_features_df.loc[valid_idx]
        return best_features_df


# Define preprocessing pipeline
main_pipeline = Pipeline([
    ("categorical_encoder", CategoricalOneHotEncoder()),
    ("standard_scaler", Standardization()),
    ("outlier_remover", RemoveOutliersAndAddFeatures(target_labels=train_labels)),
])

# Apply pipeline to train data
processed_train_features = main_pipeline.fit_transform(train_features)
train_labels = processed_train_features["target_labels"]
processed_train_features.drop("target_labels", axis=1, inplace=True)


# Commented out: Cross-validation for baseline models
# lin_reg = LinearRegression()
# cross_scores_lin = -cross_val_score(lin_reg, processed_train_features, train_labels, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
# print(pd.Series(cross_scores_lin).describe())  # Mean RMSE: ~2.604627e+07

# decision_tree = DecisionTreeRegressor(random_state=42)
# cross_scores_tree = -cross_val_score(decision_tree, processed_train_features, train_labels, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
# print(pd.Series(cross_scores_tree).describe())  # Mean RMSE: ~1.140864e+07

# random_forest = RandomForestRegressor(random_state=42, n_jobs=-1)
# cross_scores_rf = -cross_val_score(random_forest, processed_train_features, train_labels, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
# print(pd.Series(cross_scores_rf).describe())  # Mean RMSE: ~1.137091e+07

# gb_reg = GradientBoostingRegressor(random_state=42)
# cross_scores_gb = -cross_val_score(gb_reg, processed_train_features, train_labels, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
# print(pd.Series(cross_scores_gb).describe())  # Mean RMSE: ~2.151815e+07


# Hyperparameter tuning for GradientBoostingRegressor using RandomizedSearchCV
gb_reg = GradientBoostingRegressor(random_state=42)

param_dist = {
    'n_estimators': randint(50, 250),
    'learning_rate': uniform(0.01, 0.39),
    'max_depth': randint(3, 8),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 7),
    'subsample': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(gb_reg, n_iter=20, random_state=42, param_distributions=param_dist,
                                   scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)

random_search.fit(processed_train_features, train_labels)

# print(f"best_estimator {random_search.best_estimator_}")
# print(f"best_param {random_search.best_params_}")
# print(f"best_score {random_search.best_score_}")  # Example: R2 ~0.970, RMSE ~3464.34
best_model = random_search.best_estimator_


# Commented out: Alternative tuning for RandomForestRegressor
# random_forest = RandomForestRegressor(random_state=42, n_jobs=-1)
# param_dist_rf = {
#     "n_estimators": randint(50, 250),
#     'max_depth': randint(3, 15),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 10),
#     'max_features': uniform(0.1, 0.9)
# }
# random_search_rf = RandomizedSearchCV(random_forest, n_iter=20, random_state=42, param_distributions=param_dist_rf,
#                                      scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1, verbose=2)
# random_search_rf.fit(processed_train_features, train_labels)
# best_model = random_search_rf.best_estimator_  # Example: R2 ~0.968, RMSE ~3578.53


# Prepare test set
test_features = test_set.drop("price", axis=1)
test_labels = test_set["price"].copy()

# Extract transformers from pipeline
categorical_transformer = main_pipeline.named_steps['categorical_encoder']
standard_transformer = main_pipeline.named_steps['standard_scaler']
outlier_transformer = main_pipeline.named_steps['outlier_remover']

# Apply transformations to test data
test_processed = categorical_transformer.transform(test_features)
test_processed = standard_transformer.transform(test_processed)

# Detect and remove outliers in test data
outliers_test = outlier_transformer.isolation_forest.predict(test_processed)
test_processed = test_processed[outliers_test == 1]
test_labels = test_labels[outliers_test == 1]

# Add features to test data
poly = outlier_transformer.poly
selector = outlier_transformer.selector

new_features_test = poly.transform(test_processed)
selected_features_test = selector.transform(new_features_test)
columns = poly.get_feature_names_out()[selector.get_support()]
processed_test_features = pd.DataFrame(selected_features_test, index=test_processed.index, columns=columns)

# Remove IQR outliers from 'class duration' if present
if 'class duration' in processed_test_features.columns:
    valid_test_idx = remove_outliers_with_iqr(processed_test_features['class duration']).index
    processed_test_features = processed_test_features.loc[valid_test_idx]
    test_labels = test_labels.loc[valid_test_idx]

# Predict and evaluate on test data
test_predictions = best_model.predict(processed_test_features)

rmse = root_mean_squared_error(test_labels, test_predictions)
r2 = r2_score(test_labels, test_predictions)

print(f"RMSE on test set: {rmse}")  # Example: 4677.06
print(f"R2 Score on test set: {r2}")  # Example: 0.947

# Commented out: Visualization of predictions
# plt.scatter(test_labels, test_predictions, alpha=0.7, color='dodgerblue')
# plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], color='red', linestyle='--')
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.show()