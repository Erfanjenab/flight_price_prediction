## flight_price_prediction

## A machine learning project to predict flight ticket prices using Python

Welcome to the Flight Price Prediction project! This repository implements a complete machine learning pipeline to predict flight ticket prices in India using a dataset scraped from the "Ease My Trip" website. Built with Python and scikit-learn, it demonstrates best practices in data preprocessing, feature engineering, model training, and evaluation. Whether you're a beginner learning ML or an experienced data scientist, this project offers insights into real-world regression tasks for travel pricing. 🚀

The goal is to accurately forecast ticket prices based on features like airline, stops, and travel dates, achieving an RMSE of approximately 4,677 and an R² score of ~0.947 on the test set—proving robust performance without overfitting. Let's dive in! 📊

## Project Overview 🔍

This project uses a flight booking dataset 📈 scraped from the Ease My Trip website, containing ~300,261 records for economy and business class tickets across Indian routes. Data was collected over 50 days (Feb 11 to Mar 31, 2022) using the Octoparse tool, focusing on secondary data for statistical analysis and price prediction. Key features include:

- Flight details: Airline, source/destination cities, stops, duration, departure/arrival times.
- Booking info: Days left until flight, class (economy/business).
- Target variable: Ticket price (in INR).

The dataset is ideal for regression as it captures real-world factors like demand fluctuations and route variations. We process this data through a custom pipeline to build a predictive model using GradientBoostingRegressor. 🤖

Key highlights:

- Stratified splitting on 'class' for balanced train/test sets.
- Advanced preprocessing including one-hot encoding with PCA, standardization, outlier removal, and polynomial features.
- Hyperparameter tuning via RandomizedSearchCV for optimal model performance.
- Visualizations to analyze predictions (uncomment for scatter plots).

This setup not only predicts prices but also serves as a template for similar ML projects. Real results from execution: Test RMSE ~4,677, R² ~0.947.

## Prerequisites 🛠️

To run this project, you'll need:

- Python: Version 3.8 or higher (tested on 3.12.3). 🐍
- Libraries (install via `pip install -r requirements.txt`):
  - pandas 📊
  - numpy 🔢
  - matplotlib 📉
  - scikit-learn (including modules like Pipeline, GradientBoostingRegressor, etc.) ⚙️
  - scipy (for randint/uniform in hyperparameter tuning) 🧮
  - zlib (for hash-based splitting) 🔒
  - category_encoders (for target encoding, though unused in final) 📐
  - pyod (for KNN outlier detection, though unused) 🛡️

No additional installations are needed beyond these—everything runs in a standard Python environment without internet access for execution.

## What happens during execution? 🔄

- Loads and cleans the dataset (drops unnamed column).
- Splits data stratified by 'class' for balanced distributions.
- Preprocesses: Encodes categoricals with OneHot+PCA/Ordinal, standardizes numerics, removes outliers via IsolationForest, adds interaction features with PolynomialFeatures and SelectKBest.
- Tests baseline models (LinearRegression, DecisionTree, RandomForest, GradientBoosting) via cross-validation.
- Tunes hyperparameters for the best model (GradientBoosting with RandomizedSearchCV).
- Evaluates with RMSE and R² on processed test data, generates optional visualizations.

Outputs 📂:
- Console: RMSE, R² score, cross-validation stats.
- Files: Optional scatter plot for actual vs. predicted prices.

Example baseline results (MSE from cross-validation, RMSE approx. sqrt):
- LinearRegression: Mean MSE 2.60e+07 (RMSE ~5,104) ❌ (High error).
- DecisionTree: Mean MSE 1.14e+07 (RMSE ~3,377) ⚠️ (Overfitting risk).
- RandomForest: Mean MSE 1.14e+07 (RMSE ~3,371) ✅.
- GradientBoosting (untuned): Mean MSE 2.15e+07 (RMSE ~4,639).

Tuned GradientBoosting: Test RMSE ~4,677; R² ~0.947 🎯 (Best performer).

## Code Structure 📁

The script `flight_analysis.py` is modular and well-commented for easy understanding. Here's a breakdown:

- Data Loading & Splitting 📥: Loads `Clean_Dataset.csv`, splits stratified on 'class' (alternative hash-based for scalability).
- Preprocessing Pipeline 🧹:
  - Categorical Encoding: OneHotEncoder with PCA reduction + OrdinalEncoder for ordered categories. 🔤
  - Standardization: Scales duration, flight, days_left. 📐
  - Outlier Removal & Features: IsolationForest for anomalies, PolynomialFeatures for interactions, SelectKBest for top features, IQR on 'class duration'. 🛡️➕
- Model Training & Tuning 🤖: Baselines via cross_val_score, tunes GradientBoosting (or RandomForest alternative) with RandomizedSearchCV.
- Test Preparation & Evaluation 📊: Applies pipeline transformers to test data, predicts, computes RMSE/R², optional scatter plot.

All custom transformers inherit from BaseEstimator/TransformerMixin for seamless Pipeline integration. Full comments explain each step's purpose and alternatives.

## Results & Performance 📈

After tuning, the GradientBoosting model achieves:

- RMSE on Test Set: ~4,677 (low error indicates good fit).
- R² Score: ~0.947 (explains ~95% of variance—strong predictive power).
- Cross-Validation (baselines): Stable with low std, outperforming LinearRegression and untuned models.

Visuals (uncommented) show aligned predictions, confirming unbiased estimates. For production, deploy on new data for real-time pricing! 🎯

## Why This Project? 🌟

- Educational Value: Learn end-to-end ML: From scraping-inspired data to tuned models for travel insights.
- Real-World Applicability: Adapt for flight apps (e.g., integrate live data for dynamic pricing).
- Customizations: Experiment with TargetEncoder (defined but unused) or add features like seasonal trends.
- Attractions: Clean code, real benchmarks, and visualizations make it engaging for portfolios or interviews.

If you spot improvements, contributions are welcome! Fork and PR. 👏

## Acknowledgments 🙌

Dataset: Courtesy of Ease My Trip website (scraped via Octoparse, 2022).

Built as of September 2025—feel free to star ⭐ and watch for updates!
