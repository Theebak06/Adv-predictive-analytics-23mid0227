# Generated from: 23MID0227_LAB01_HOUSEPRICE.ipynb
# Converted at: 2026-07-20T16:20:54.721Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#NAME:THEEBAK S
#REG NO:23MID0227

#DATASET A[rEAL ESTATE DATASET]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. SETUP & EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
SEED = 42
np.random.seed(SEED)

df_a = pd.read_excel('C:/Users/ADMIN/Downloads/real+estate+valuation+data+set/Real estate valuation data set.xlsx')
target_a = 'Y house price of unit area'
X_a = df_a.drop(columns=[target_a, 'No'])
y_a = df_a[target_a]

# Save EDA Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_a[target_a], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Target Variable Distribution (Dataset A)')
sns.heatmap(df_a.drop(columns=['No']).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
axes[1].set_title('Feature Correlation Matrix (Dataset A)')
plt.tight_layout()
plt.savefig('dataset_a_eda_plots.png')
plt.close()

# ==========================================
# 2. LEAKAGE-SAFE PREPROCESSING & SPLIT
# ==========================================
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.20, random_state=SEED)
pre_a = ColumnTransformer([('num', StandardScaler(), X_a.columns.tolist())])

# Storage dict for final structural metrics comparison table
evaluation_summary_a = {}

def track_metrics(name, pipeline, num_features_used):
    pipeline.fit(X_train_a, y_train_a)
    preds = pipeline.predict(X_test_a)
    
    mae = mean_absolute_error(y_test_a, preds)
    rmse = np.sqrt(mean_squared_error(y_test_a, preds))
    r2 = r2_score(y_test_a, preds)
    
    # Calculate dimensional adjusted R2 bounds manually
    n = len(y_test_a)
    p = num_features_used
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    
    evaluation_summary_a[name] = [mae, rmse, r2, adj_r2]

# ==========================================
# 3. CORE REGRESSION MODEL BENCHMARKS
# ==========================================
# Model 1: Simple Linear Regression (1 predictor feature)
pipe_slr = Pipeline([
    ('select', ColumnTransformer([('num', StandardScaler(), ['X3 distance to the nearest MRT station'])])),
    ('model', LinearRegression())
])
track_metrics("Simple Linear Regression", pipe_slr, num_features_used=1)

# Model 2: Multiple Linear Regression (6 baseline numerical columns)
pipe_mlr = Pipeline([('preprocess', pre_a), ('model', LinearRegression())])
track_metrics("Multiple Linear Regression", pipe_mlr, num_features_used=6)

# Model 3: Polynomial Regression (Degree 2 creates 27 structural features)
pipe_poly = Pipeline([
    ('preprocess', pre_a),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])
track_metrics("Polynomial Regression (D2)", pipe_poly, num_features_used=27)

# Model 4: Random Forest Hyperparameter Cross-Validation Tuning
pipe_rf = Pipeline([('preprocess', pre_a), ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [6, 10],
    'model__min_samples_leaf': [2, 4]
}

grid_search_a = GridSearchCV(pipe_rf, param_grid, cv=cv_strategy, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search_a.fit(X_train_a, y_train_a)

# Score optimized ensemble output
track_metrics("Random Forest (Tuned)", grid_search_a.best_estimator_, num_features_used=6)

# ==========================================
# 4. REPORT METRICS SUMMARY TABLE
# ==========================================
df_metrics_a = pd.DataFrame.from_dict(
    evaluation_summary_a, orient='index', columns=['MAE', 'RMSE', 'R2 Score', 'Adjusted R2']
)
print("\n=== FINAL TEST EVALUATION METRICS (DATASET A) ===")
print(df_metrics_a.round(4))

print("\n=== GRID SEARCH HYPERPARAMETER TUNING ===")
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 6, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search_a = GridSearchCV(pipe_rf, param_grid, cv=cv_strategy, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search_a.fit(X_train_a, y_train_a)

print(f"Optimal Parameters: {grid_search_a.best_params_}")
print(f"Best CV RMSE Score: {-grid_search_a.best_score_:.4f}")

# Evaluate the optimized model on final holdout data
def evaluate_pipeline_a(name, pipeline):
    pipeline.fit(X_train_a, y_train_a)
    preds = pipeline.predict(X_test_a)
    mae = mean_absolute_error(y_test_a, preds)
    rmse = np.sqrt(mean_squared_error(y_test_a, preds))
    r2 = r2_score(y_test_a, preds)
    print(f"{name:<28} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    return preds
final_preds_a = evaluate_pipeline_a("Random Forest (Tuned Final)", grid_search_a.best_estimator_)



#DATASET B[AMES HOUSING DATASET]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. SETUP & REPRODUCIBILITY CONTROLS
# ==========================================
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# ==========================================
# 2. DATA ACQUISITION & OUTLIER MITIGATION
# ==========================================
df_ames = pd.read_csv("C:/Users/ADMIN/Downloads/archive (9)/AmesHousing.csv")

# Drop documented high-variance luxury outliers (>4000 sq ft) to improve baseline MAE
df_clean = df_ames[df_ames['Gr Liv Area'] < 4000].copy()

X = df_clean.drop(columns=['SalePrice', 'Order', 'PID'])
y = df_clean['SalePrice']

# ==========================================
# 3. COMPREHENSIVE VISUALIZATIONS (EDA)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot A: Target SalePrice Distribution Profile
sns.histplot(df_clean['SalePrice'], kde=True, ax=axes[0, 0], color='royalblue')
axes[0, 0].set_title('Figure 1: Target SalePrice Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('SalePrice ($)')

# Plot B: Target vs Ground Living Area Relationship
sns.scatterplot(data=df_clean, x='Gr Liv Area', y='SalePrice', ax=axes[0, 1], color='darkorange', alpha=0.5)
axes[0, 1].set_title('Figure 2: SalePrice vs Ground Living Area', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Gr Liv Area (sq ft)')
axes[0, 1].set_ylabel('SalePrice ($)')

# Plot C: Valuation by Overall Material Grade Bracket
sns.boxplot(data=df_clean, x='Overall Qual', y='SalePrice', ax=axes[1, 0], palette='Blues')
axes[1, 0].set_title('Figure 3: SalePrice by Overall Quality Grade', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Overall Quality (1-10)')
axes[1, 0].set_ylabel('SalePrice ($)')

# Plot D: Statistical Feature Correlation Heatmap
top_numeric = df_clean.select_dtypes(include=np.number).corr()['SalePrice'].sort_values(ascending=False).index[:8]
sns.heatmap(df_clean[top_numeric].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title('Figure 4: Correlation Matrix of Top Predictors', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ames_eda_comprehensive.png', dpi=300)
plt.show()

# ==========================================
# 4. LEAKAGE-SAFE PREPROCESSING & HOLDOUT DESIGN
# ==========================================
# Split data into training and test partitions before extraction of transformation metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

def get_adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

# Global transformation pipeline wrapper
preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
])

results = {}

# =======================================================
# MODEL 1: SIMPLE LINEAR REGRESSION
# =======================================================
pipe_slr = Pipeline([
    ('select', ColumnTransformer([('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Overall Qual'])])),
    ('model', LinearRegression())
])
pipe_slr.fit(X_train, y_train)
p_slr = pipe_slr.predict(X_test)
r2_slr = r2_score(y_test, p_slr)
results['Simple Linear Regression'] = [mean_absolute_error(y_test, p_slr), np.sqrt(mean_squared_error(y_test, p_slr)), r2_slr, get_adjusted_r2(r2_slr, len(y_test), 1)]

# =======================================================
# MODEL 2: MULTIPLE LINEAR REGRESSION
# =======================================================
pipe_mlr = Pipeline([
    ('preprocess', ColumnTransformer([('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features)])),
    ('model', LinearRegression())
])
pipe_mlr.fit(X_train, y_train)
p_mlr = pipe_mlr.predict(X_test)
r2_mlr = r2_score(y_test, p_mlr)
results['Multiple Linear Regression'] = [mean_absolute_error(y_test, p_mlr), np.sqrt(mean_squared_error(y_test, p_mlr)), r2_mlr, get_adjusted_r2(r2_mlr, len(y_test), len(num_features))]

# =======================================================
# MODEL 3: POLYNOMIAL REGRESSION (DEGREE 2)
# =======================================================
poly_features = ['Overall Qual', 'Gr Liv Area', 'Total Bsmt SF']
pre_poly = ColumnTransformer([
    ('poly_num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler())]), poly_features)
])
pipe_poly = Pipeline([('preprocess', pre_poly), ('model', LinearRegression())])
pipe_poly.fit(X_train, y_train)
p_poly = pipe_poly.predict(X_test)
r2_poly = r2_score(y_test, p_poly)
p_poly_num = pre_poly.fit_transform(X_train).shape[1]
results['Polynomial Regression (D2)'] = [mean_absolute_error(y_test, p_poly), np.sqrt(mean_squared_error(y_test, p_poly)), r2_poly, get_adjusted_r2(r2_poly, len(y_test), p_poly_num)]

# =======================================================
# MODEL 4: TUNED RANDOM FOREST REGRESSOR (HYPERPARAMETER OPTIMIZATION)
# =======================================================
pipe_rf = Pipeline([('preprocess', preprocess), ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
cv_strat = KFold(n_splits=3, shuffle=True, random_state=SEED)

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [20, 25],
    'model__max_features': [0.3, 0.4]
}

grid_rf = GridSearchCV(pipe_rf, param_grid, cv=cv_strat, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
p_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, p_rf)
p_total = preprocess.fit_transform(X_train).shape[1]
results['Random Forest (Tuned Final)'] = [mean_absolute_error(y_test, p_rf), np.sqrt(mean_squared_error(y_test, p_rf)), r2_rf, get_adjusted_r2(r2_rf, len(y_test), p_total)]

# ==========================================
# 5. TECHNICAL METRICS COMPILATION TABLE & CONSOLE REPORT
# ==========================================
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['MAE', 'RMSE', 'R2 Score', 'Adjusted R2'])
print("\n=== AMES HOUSING COMPREHENSIVE PERFORMANCE PROFILE ===")
print(df_results.round(4).to_string())
print(f"\nBest Tuning Parameters Configured: {grid_rf.best_params_}")

# ==========================================
# 6. ENSEMBLE MODEL DIAGNOSTIC PLOTS
# ==========================================
fig_diag, axes_diag = plt.subplots(1, 2, figsize=(13, 5))

# Plot E: Actual vs Predicted Pricing Check
axes_diag[0].scatter(y_test, p_rf, alpha=0.5, color='forestgreen')
lo, hi = min(y_test.min(), p_rf.min()), max(y_test.max(), p_rf.max())
axes_diag[0].plot([lo, hi], [lo, hi], 'k--', lw=2)
axes_diag[0].set_title('Figure 5: Actual vs Predicted (Tuned Random Forest)', fontsize=11, fontweight='bold')
axes_diag[0].set_xlabel('Actual SalePrice ($)')
axes_diag[0].set_ylabel('Predicted SalePrice ($)')

# Plot F: Symmetrical Residual Error Distribution
residuals = y_test - p_rf
axes_diag[1].scatter(p_rf, residuals, alpha=0.5, color='purple')
axes_diag[1].axhline(0, color='black', linestyle='--', lw=2)
axes_diag[1].set_title('Figure 6: Residual Error Scatter Plot', fontsize=11, fontweight='bold')
axes_diag[1].set_xlabel('Predicted SalePrice ($)')
axes_diag[1].set_ylabel('Residual Error ($)')

plt.tight_layout()
plt.savefig('ames_model_diagnostics.png', dpi=300)
plt.show()