# %%
"""
# Comprehensive EDA & Machine Learning Analysis

## Complete Data Science Workflow
This notebook demonstrates a systematic approach to exploratory data analysis (EDA) 
and machine learning using medical insurance data. The workflow covers:

- Data loading, cleaning, and preprocessing
- Statistical analysis and visualization techniques  
- Correlation analysis and hypothesis testing
- Multiple machine learning models with evaluation
- Advanced Python techniques for data science

**Dataset**: Medical Insurance Charges (1,338 records, 7 features)
**Goal**: Predict insurance charges and understand key factors
"""

# %%
"""
## Learning Objectives

### Data Science Fundamentals
- **Data Exploration**: Descriptive statistics, data types, missing value handling
- **Statistical Analysis**: Correlation testing, ANOVA, chi-square tests
- **Data Visualization**: Distribution plots, relationship analysis, correlation heatmaps

### Machine Learning Pipeline  
- **Model Implementation**: Linear, Polynomial, Ridge, Decision Tree, Random Forest
- **Model Evaluation**: R², MSE, MAE, RMSE metrics with cross-validation
- **Hyperparameter Tuning**: GridSearchCV with systematic optimization

### Advanced Techniques
- **Statistical Methods**: f_oneway, pearsonr, chi2_contingency, confidence intervals
- **Python Skills**: np.polyfit, np.where, pd.get_dummies, regex patterns
- **Production Tools**: Pipelines, progress tracking, memory optimization
"""

# %%
# IMPORT LIBRARIES
# Core data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Statistical analysis libraries
from scipy import stats
from scipy.stats import f_oneway, pearsonr, chi2_contingency, chi2

# Utility libraries
from tqdm import tqdm
import requests
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Libraries imported successfully")
print("Available components: Data processing, ML models, Statistical tests, Visualization")

# %%
"""
## PART 1: DATA LOADING AND EXPLORATION

### Objectives:
- Load medical insurance dataset from web source
- Examine dataset structure, dimensions, and data types
- Perform initial data quality assessment
- Demonstrate advanced pandas inspection techniques
"""

# %%
# DATA LOADING
print("Loading Medical Insurance Dataset...")

# Load dataset from web source
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), header=None)

# Set descriptive column names
columns = ['age', 'gender', 'bmi', 'children', 'smoker', 'region', 'charges']
df.columns = columns

# Initial dataset inspection
print(f"Dataset loaded successfully")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Target variable: charges (insurance cost prediction)")

# Display sample data
df.head()

# %%
"""
## PART 2: DATA CLEANING AND PREPROCESSING

### Objectives:
- Identify and handle missing values using appropriate strategies
- Convert data types for proper analysis
- Create encoded variables for categorical data
- Demonstrate advanced data selection and inspection methods
"""

# %%
# DATA CLEANING AND PREPROCESSING
print("Starting data cleaning process...")

# Handle missing values (replace '?' with NaN)
df.replace('?', np.nan, inplace=True)

# Missing value analysis
print("\nMissing Values Assessment:")
missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentages
}).round(2)
print(missing_df)

# Data type conversion with error handling
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

# Missing value imputation strategies
df['age'].fillna(df['age'].mean(), inplace=True)  # Mean imputation for age
df['bmi'].fillna(df['bmi'].mean(), inplace=True)  # Mean imputation for BMI
df['smoker'].fillna(df['smoker'].mode()[0], inplace=True)  # Mode imputation for categorical

# Create binary encoded variables for machine learning
df['smoker_encoded'] = df['smoker'].map({'yes': 1, 'no': 0})
df['gender_encoded'] = df['gender'].map({'male': 1, 'female': 0})

print(f"\nData cleaning completed")
print(f"Final dataset shape: {df.shape}")

# Advanced data inspection techniques
print("\nData Types Analysis:")
print(df.dtypes)
print("\nDataset Information:")
print(df.info())

# Comprehensive data quality assessment
print("\nData Quality Metrics:")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Unique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()}")

# Advanced pandas selection techniques
print("\nData Selection Methods:")
print("Numerical columns:", df.select_dtypes(include=['number']).columns.tolist())
print("Categorical columns:", df.select_dtypes(include=['object']).columns.tolist())
print("Alternative numeric selection:", df._get_numeric_data().columns.tolist())

# %%
"""
## PART 3: DESCRIPTIVE STATISTICS

### Objectives:
- Calculate central tendency and variability measures
- Analyze distributions of numerical variables
- Examine categorical variable frequencies and percentages
- Understand data characteristics before modeling
"""

# %%
# DESCRIPTIVE STATISTICS ANALYSIS
print("DESCRIPTIVE STATISTICS ANALYSIS")
print("="*50)

# Numerical variables analysis
numerical_stats = df.describe()
print("\nNumerical Variables Summary:")
print(numerical_stats.round(2))

# Key insights from numerical summary
print("\nKey Numerical Insights:")
print(f"Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
print(f"BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
print(f"Charges range: ${df['charges'].min():.0f} - ${df['charges'].max():.0f}")
print(f"Average charges: ${df['charges'].mean():.0f}")

# Categorical variables analysis  
print("\nCategorical Variables Summary:")
categorical_cols = ['gender', 'smoker', 'region']
for col in categorical_cols:
    print(f"\n{col.upper()} Distribution:")
    value_counts = df[col].value_counts()
    percentages = (value_counts / len(df) * 100).round(1)
    summary_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages
    })
    print(summary_df)

# %%
"""
## PART 4: DATA VISUALIZATION

### Objectives:
- Create distribution plots to understand variable shapes
- Examine relationships between variables using scatter plots
- Compare groups using box plots and violin plots
- Apply regression analysis with confidence intervals
- Demonstrate multiple matplotlib and seaborn techniques
"""

# %%
# COMPREHENSIVE VISUALIZATION ANALYSIS
print("Creating comprehensive visualization suite...")

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 24))

# 1. Distribution plots
plt.subplot(4, 3, 1)
plt.hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(4, 3, 2)
plt.hist(df['bmi'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('BMI Distribution', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Frequency')

plt.subplot(4, 3, 3)
plt.hist(df['charges'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
plt.title('Charges Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Charges ($)')
plt.ylabel('Frequency')

# 2. Box plots
plt.subplot(4, 3, 4)
sns.boxplot(data=df, x='smoker', y='charges', palette='Set2')
plt.title('Charges by Smoking Status', fontsize=14, fontweight='bold')
plt.xlabel('Smoker')
plt.ylabel('Charges ($)')

plt.subplot(4, 3, 5)
sns.boxplot(data=df, x='gender', y='charges', palette='Set1')
plt.title('Charges by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Charges ($)')

plt.subplot(4, 3, 6)
sns.boxplot(data=df, x='region', y='charges', palette='viridis')
plt.title('Charges by Region', fontsize=14, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Charges ($)')
plt.xticks(rotation=45)

# 3. Scatter plots
plt.subplot(4, 3, 7)
plt.scatter(df['age'], df['charges'], alpha=0.6, c=df['smoker_encoded'], cmap='coolwarm')
plt.title('Age vs Charges (colored by smoking)', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Charges ($)')
plt.colorbar(label='Smoker (1=Yes, 0=No)')

plt.subplot(4, 3, 8)
plt.scatter(df['bmi'], df['charges'], alpha=0.6, c=df['children'], cmap='plasma')
plt.title('BMI vs Charges (colored by children)', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')
plt.colorbar(label='Number of Children')

# 4. Regression plots
plt.subplot(4, 3, 9)
sns.regplot(data=df, x='age', y='charges', scatter_kws={'alpha':0.6})
plt.title('Age vs Charges (with regression line)', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Charges ($)')

plt.subplot(4, 3, 10)
sns.regplot(data=df, x='bmi', y='charges', scatter_kws={'alpha':0.6}, color='green')
plt.title('BMI vs Charges (with regression line)', fontsize=14, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')

# 5. KDE plots
plt.subplot(4, 3, 11)
sns.kdeplot(data=df, x='charges', hue='smoker', fill=True, alpha=0.6)
plt.title('Charges Distribution by Smoking Status (KDE)', fontsize=14, fontweight='bold')
plt.xlabel('Charges ($)')
plt.ylabel('Density')

# 6. Violin plot
plt.subplot(4, 3, 12)
sns.violinplot(data=df, x='smoker', y='charges', palette='muted')
plt.title('Charges Distribution by Smoking Status (Violin)', fontsize=14, fontweight='bold')
plt.xlabel('Smoker')
plt.ylabel('Charges ($)')

plt.tight_layout()
plt.show()

# %%
"""
## PART 5: CORRELATION ANALYSIS

### Objectives:
- Calculate correlation coefficients between numerical variables
- Create correlation heatmaps with multiple styling approaches
- Test statistical significance of correlations
- Implement advanced matrix operations using numpy
- Identify strongest predictors of insurance charges
"""

# %%
# CORRELATION ANALYSIS
print("Performing correlation analysis...")

# Select numerical variables for correlation analysis
numerical_cols = ['age', 'bmi', 'children', 'charges', 'smoker_encoded', 'gender_encoded']
corr_df = df[numerical_cols]

# Calculate correlation matrix
correlation_matrix = corr_df.corr()

# Create multiple heatmaps with different visualization styles
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Heatmap 1: Classic style
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', linewidths=0.5, ax=axes[0,0])
axes[0,0].set_title('Correlation Heatmap - Classic Style', fontsize=16, fontweight='bold')

# Heatmap 2: Diverging colormap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', linewidths=1, ax=axes[0,1])
axes[0,1].set_title('Correlation Heatmap - Cool-Warm Style', fontsize=16, fontweight='bold')

# Heatmap 3: Masked upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='viridis', 
            center=0, square=True, fmt='.3f', linewidths=0.5, ax=axes[1,0])
axes[1,0].set_title('Correlation Heatmap - Lower Triangle', fontsize=16, fontweight='bold')

# Heatmap 4: Custom colors
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, fmt='.3f', linewidths=0.5, 
            cbar_kws={'shrink': 0.8}, ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap - Red-Yellow-Blue Style', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# Print strongest correlations
print("\nStrongest Correlations with Target Variable (Charges):")
charges_corr = correlation_matrix['charges'].abs().sort_values(ascending=False)[1:]
for var, corr in charges_corr.items():
    print(f"  {var}: {corr:.3f}")

# Statistical significance testing for correlations
print("\nStatistical Significance Testing:")
for col in numerical_cols:
    if col != 'charges':
        corr_coef, p_val = pearsonr(df[col].dropna(), df['charges'].dropna())
        
        # Determine significance level
        if abs(corr_coef) > 0.7 and p_val < 0.001:
            significance = "Strong***"
        elif abs(corr_coef) > 0.3 and p_val < 0.05:
            significance = "Moderate**"
        elif p_val < 0.1:
            significance = "Weak*"
        else:
            significance = "None"
            
        print(f"  {col:15} | r = {corr_coef:6.3f} | p = {p_val:.6f} | {significance}")

# Advanced numpy matrix operations
print("\nAdvanced Matrix Operations:")

# Upper triangle masking technique (np.triu)
mask_upper = np.triu(np.ones_like(correlation_matrix, dtype=bool))
print(f"  Upper triangle mask shape: {mask_upper.shape}")

# Lower triangle masking technique (np.tril)
mask_lower = np.tril(np.ones_like(correlation_matrix, dtype=bool))
print(f"  Lower triangle mask shape: {mask_lower.shape}")

# Function to extract top correlation pairs
def get_top_correlations(corr_matrix, n=5):
    """Extract top n correlation pairs from correlation matrix"""
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_tri.unstack().dropna()
    return correlations.abs().sort_values(ascending=False).head(n)

top_pairs = get_top_correlations(correlation_matrix)
print(f"\nTop {len(top_pairs)} Correlated Variable Pairs:")
for (var1, var2), corr in top_pairs.items():
    print(f"  {var1} ↔ {var2}: {corr:.3f}")

# %%
"""
## PART 6: INTERACTIVE PLOTLY VISUALIZATIONS

### Objectives:
- Create interactive scatter plots with multiple dimensions
- Build 3D visualizations for complex data relationships
- Implement hover data and dynamic filtering
- Demonstrate advanced Plotly features and customization
- Compare static vs interactive visualization approaches
"""

# %%
# INTERACTIVE PLOTLY VISUALIZATIONS
print("Creating interactive Plotly visualizations...")

# 1. Multi-dimensional interactive scatter plot
fig1 = px.scatter(df, x='age', y='charges', color='smoker', size='bmi',
                 hover_data=['gender', 'children', 'region'],
                 title='Interactive Scatter Plot: Age vs Charges',
                 labels={'charges': 'Charges ($)', 'age': 'Age (years)'},
                 color_discrete_map={'yes': 'red', 'no': 'blue'})
fig1.update_layout(height=600, showlegend=True)
fig1.show()

# 2. 3D scatter plot for complex relationships
fig2 = px.scatter_3d(df, x='age', y='bmi', z='charges', color='smoker',
                    size='children', hover_data=['gender', 'region'],
                    title='3D Interactive Plot: Age, BMI, and Charges',
                    labels={'charges': 'Charges ($)', 'age': 'Age (years)', 'bmi': 'BMI'})
fig2.update_layout(height=700)
fig2.show()

# 3. Interactive box plot with grouping
fig3 = px.box(df, x='smoker', y='charges', color='gender',
              title='Interactive Box Plot: Charges by Smoking Status and Gender',
              labels={'charges': 'Charges ($)', 'smoker': 'Smoking Status'})
fig3.update_layout(height=500)
fig3.show()

# 4. Interactive correlation heatmap
fig4 = px.imshow(correlation_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                title='Interactive Correlation Heatmap',
                labels=dict(color="Correlation"))
fig4.update_layout(height=600)
fig4.show()

# 5. Custom interactive plot with Go.Figure
if 'age' in df.columns:
    # Create advanced plot using graph objects
    fig5 = go.Figure()
    
    # Add traces for different smoker categories
    for smoker_status in df['smoker'].unique():
        subset = df[df['smoker'] == smoker_status]
        fig5.add_trace(go.Scatter(
            x=subset['age'],
            y=subset['charges'],
            mode='markers',
            name=f'Smoker: {smoker_status}',
            hovertemplate='<b>Age:</b> %{x}<br><b>Charges:</b> $%{y:,.0f}<br><b>Smoker:</b> ' + smoker_status,
            marker=dict(size=8, opacity=0.6)
        ))
    
    fig5.update_layout(
        title='Interactive Age vs Charges by Smoking Status',
        xaxis_title='Age',
        yaxis_title='Charges ($)',
        hovermode='closest',
        height=500
    )
    fig5.show()

# Advanced Plotly capabilities demonstrated
print("Advanced Plotly Features Demonstrated:")
print("  Multi-dimensional scatter plots with size and color encoding")
print("  3D visualizations for exploring complex relationships")
print("  Interactive hover data with custom formatting")
print("  Dynamic color mapping and legends")
print("  Responsive layout optimization")
print("  Custom graph objects for advanced control")

# %%
"""
## PART 7: MACHINE LEARNING MODELS

### Objectives:
- Implement multiple regression algorithms (Linear, Polynomial, Ridge, Tree-based)
- Compare model performance using multiple evaluation metrics
- Demonstrate proper train/test split methodology
- Apply scikit-learn pipeline and preprocessing techniques
- Evaluate model performance and identify best performer
"""

# %%
# MACHINE LEARNING MODEL IMPLEMENTATION
print("Building and evaluating machine learning models...")

# Feature selection and target definition
features = ['age', 'bmi', 'children', 'smoker_encoded', 'gender_encoded']
X = df[features]
y = df['charges']

# Train-test split with proper random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Feature dimensions: {X.shape[1]} features")

# Model dictionary with diverse algorithms
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression (degree 2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Training and evaluation loop
results = {}
predictions = {}

print("\nTraining and evaluating models...")
for name, model in tqdm(models.items(), desc="Training Models"):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate comprehensive metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Store comprehensive results
    results[name] = {
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train RMSE': np.sqrt(train_mse),
        'Test RMSE': np.sqrt(test_mse)
    }
    
    predictions[name] = y_pred_test

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.round(3))

# %%
"""
<div class="alert alert-info" style="background: #e3f2fd; border-left: 5px solid #2196f3; padding: 20px; margin: 20px 0;">
    <h3 style="color: #1565c0; margin-top: 0;">📊 Step 8: Model Evaluation & Comparison</h3>
    <p style="font-size: 1.1em; margin: 0;">
        Visualize model performance and compare predictions
    </p>
</div>
"""

# %%
# 📊 MODEL EVALUATION AND COMPARISON
print("📊 Creating model evaluation visualizations...")

# 1. Model performance comparison
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# R² Score comparison
r2_scores = results_df[['Train R²', 'Test R²']]
r2_scores.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('R² Score Comparison', fontsize=16, fontweight='bold')
axes[0,0].set_ylabel('R² Score')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].legend()

# RMSE comparison
rmse_scores = results_df[['Train RMSE', 'Test RMSE']]
rmse_scores.plot(kind='bar', ax=axes[0,1], color=['lightgreen', 'salmon'])
axes[0,1].set_title('RMSE Comparison', fontsize=16, fontweight='bold')
axes[0,1].set_ylabel('RMSE')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].legend()

# MAE comparison
mae_scores = results_df[['Train MAE', 'Test MAE']]
mae_scores.plot(kind='bar', ax=axes[1,0], color=['gold', 'purple'])
axes[1,0].set_title('MAE Comparison', fontsize=16, fontweight='bold')
axes[1,0].set_ylabel('MAE')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].legend()

# Predicted vs Actual for best model
best_model = results_df.loc[results_df['Test R²'].idxmax()]
best_model_name = results_df['Test R²'].idxmax()
best_predictions = predictions[best_model_name]

axes[1,1].scatter(y_test, best_predictions, alpha=0.6, color='blue')
axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1,1].set_title(f'Predicted vs Actual - {best_model_name}', fontsize=16, fontweight='bold')
axes[1,1].set_xlabel('Actual Charges')
axes[1,1].set_ylabel('Predicted Charges')

plt.tight_layout()
plt.show()

print(f"\nBest Performing Model: {best_model_name}")
print(f"Test R² Score: {best_model['Test R²']:.4f}")
print(f"Test RMSE: ${best_model['Test RMSE']:.2f}")
print("Model selected based on highest test R² score with reasonable generalization")

# %%
"""
## PART 8: HYPERPARAMETER TUNING WITH GRIDSEARCHCV

### Objectives:
- Optimize model performance using systematic parameter search
- Implement cross-validation for robust model selection
- Compare performance before and after hyperparameter tuning
- Demonstrate GridSearchCV with multiple parameter combinations
- Analyze overfitting vs underfitting through validation curves
"""

# %%
# HYPERPARAMETER TUNING WITH GRIDSEARCHCV
print("Performing systematic hyperparameter tuning with GridSearchCV...")

# Define comprehensive parameter grids
param_grids = {
    'Ridge Regression': {
        'alpha': [0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Models for hyperparameter tuning
tuning_models = {
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Perform systematic hyperparameter tuning
tuned_results = {}
best_models = {}

for name, model in tuning_models.items():
    print(f"\nTuning {name}...")
    
    # Create GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Best model predictions
    best_pred = grid_search.best_estimator_.predict(X_test)
    
    # Store comprehensive results
    tuned_results[name] = {
        'Best Score (CV)': grid_search.best_score_,
        'Best Params': grid_search.best_params_,
        'Test R²': r2_score(y_test, best_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, best_pred)),
        'Test MAE': mean_absolute_error(y_test, best_pred)
    }
    
    best_models[name] = grid_search.best_estimator_

# Display comprehensive tuning results
print("\nHyperparameter Tuning Results:")
print("="*60)
for name, results in tuned_results.items():
    print(f"\n{name}:")
    print(f"  Best CV Score: {results['Best Score (CV)']:.4f}")
    print(f"  Test R²: {results['Test R²']:.4f}")
    print(f"  Test RMSE: ${results['Test RMSE']:.2f}")
    print(f"  Best Parameters: {results['Best Params']}")

# Create detailed results for visualization
print("\nCreating hyperparameter tuning visualization...")

# Detailed analysis of tuning results
detailed_results = []
for name, model in tuning_models.items():
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    # Extract results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['model'] = name
    detailed_results.append(cv_results)

# Combine all results
all_results = pd.concat(detailed_results, ignore_index=True)

# Create visualization for Ridge regression parameter grid
ridge_results = all_results[all_results['model'] == 'Ridge Regression']
if not ridge_results.empty:
    pivot_ridge = ridge_results.pivot_table(
        values='mean_test_score',
        index='param_alpha',
        columns='param_fit_intercept'
    )
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot_ridge, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Ridge: Alpha vs Fit_Intercept Parameter Grid')
    
    # Learning curve style plot
    plt.subplot(2, 2, 2)
    alpha_values = ridge_results['param_alpha'].unique()
    mean_scores = ridge_results.groupby('param_alpha')['mean_test_score'].mean()
    std_scores = ridge_results.groupby('param_alpha')['mean_test_score'].std()
    
    plt.errorbar(range(len(alpha_values)), mean_scores, yerr=std_scores, 
                 capsize=5, marker='o')
    plt.xticks(range(len(alpha_values)), alpha_values)
    plt.xlabel('Alpha Parameter')
    plt.ylabel('Mean CV Score')
    plt.title('Ridge: Alpha Parameter Performance')
    plt.grid(True, alpha=0.3)

# Advanced performance analysis
print("\nAdvanced Performance Analysis:")
for name, results in tuned_results.items():
    print(f"\n{name} Analysis:")
    print(f"  Cross-validation score: {results['Best Score (CV)']:.4f}")
    print(f"  Test score: {results['Test R²']:.4f}")
    print(f"  Generalization gap: {abs(results['Best Score (CV)'] - results['Test R²']):.4f}")
    
    if abs(results['Best Score (CV)'] - results['Test R²']) > 0.05:
        print(f"  Warning: Large gap suggests possible overfitting")
    else:
        print(f"  Good generalization performance")

plt.tight_layout()
plt.show()

# %%
"""
## PART 9: CROSS-VALIDATION AND MODEL SELECTION

### Objectives:
- Perform robust k-fold cross-validation for model selection
- Compare cross-validation scores across different models
- Visualize performance distributions to assess model stability
- Select final model based on comprehensive evaluation criteria
- Analyze variance in model performance across different data splits
"""

# %%
# CROSS-VALIDATION AND FINAL MODEL SELECTION
print("Performing comprehensive cross-validation for final model selection...")

# Perform cross-validation on optimized models
cv_results = {}
cv_scores = {}

for name, model in best_models.items():
    print(f"\nCross-validating {name}...")
    
    # Perform 10-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
    
    cv_results[name] = {
        'Mean CV Score': scores.mean(),
        'Std CV Score': scores.std(),
        'Min CV Score': scores.min(),
        'Max CV Score': scores.max()
    }
    
    cv_scores[name] = scores

# Create cross-validation results DataFrame
cv_df = pd.DataFrame(cv_results).T
print("\nCross-Validation Results:")
print(cv_df.round(4))

# Visualize cross-validation results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot of CV scores to show distribution
cv_data = [cv_scores[name] for name in cv_scores.keys()]
axes[0].boxplot(cv_data, labels=cv_scores.keys())
axes[0].set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=45)

# Bar plot of mean CV scores with error bars
means = [cv_results[name]['Mean CV Score'] for name in cv_results.keys()]
stds = [cv_results[name]['Std CV Score'] for name in cv_results.keys()]
axes[1].bar(range(len(means)), means, yerr=stds, capsize=5, color='skyblue', alpha=0.7)
axes[1].set_title('Mean Cross-Validation Scores with Standard Deviation', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Mean R² Score')
axes[1].set_xticks(range(len(cv_results.keys())))
axes[1].set_xticklabels(cv_results.keys(), rotation=45)

plt.tight_layout()
plt.show()

# Select final model based on cross-validation performance
final_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['Mean CV Score'])
final_model = best_models[final_model_name]

print(f"\nFinal Selected Model: {final_model_name}")
print(f"Cross-Validation Score: {cv_results[final_model_name]['Mean CV Score']:.4f} ± {cv_results[final_model_name]['Std CV Score']:.4f}")
print("Selection based on highest mean CV score with acceptable variance")

# %%
"""
## PART 10: FINAL MODEL ANALYSIS

### Objectives:
- Analyze feature importance for the selected model
- Perform comprehensive residual analysis
- Evaluate model performance on test data
- Create diagnostic plots to assess model assumptions
- Provide actionable insights from the final model
"""

# %%
# FINAL MODEL ANALYSIS
print("Analyzing final model performance and feature importance...")

# Feature importance analysis (for tree-based models)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Analysis:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title(f'Feature Importance - {final_model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

# Final model performance on test set
final_predictions = final_model.predict(X_test)
final_r2 = r2_score(y_test, final_predictions)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
final_mae = mean_absolute_error(y_test, final_predictions)

print(f"\nFinal Model Performance:")
print(f"  Test R² Score: {final_r2:.4f}")
print(f"  Test RMSE: ${final_rmse:.2f}")
print(f"  Test MAE: ${final_mae:.2f}")
print(f"  Model explains {final_r2*100:.1f}% of variance in insurance charges")

# Comprehensive residual analysis
residuals = y_test - final_predictions

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Predicted vs Actual scatter plot
axes[0,0].scatter(y_test, final_predictions, alpha=0.6)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Actual Charges ($)')
axes[0,0].set_ylabel('Predicted Charges ($)')

# Residuals vs Predicted (for homoscedasticity check)
axes[0,1].scatter(final_predictions, residuals, alpha=0.6)
axes[0,1].axhline(y=0, color='r', linestyle='--')
axes[0,1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Predicted Charges ($)')
axes[0,1].set_ylabel('Residuals')

# Histogram of residuals (for normality check)
axes[1,0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
axes[1,0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Residuals')
axes[1,0].set_ylabel('Frequency')

# Q-Q plot for normality assessment
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
"""
## PART 11: SUMMARY AND CONCLUSIONS

### Key Findings and Insights:
This comprehensive analysis provides actionable insights for insurance pricing
and risk assessment strategies based on systematic data analysis and modeling.
"""

# %%
"""
## SUMMARY AND KEY FINDINGS

### Data Exploration Insights:
1. **Dataset:** Medical insurance charges prediction with 1,338 samples
2. **Key Variables:** Age, BMI, children, smoking status, gender, region
3. **Target Variable:** Insurance charges (highly skewed distribution)
4. **Missing Values:** Successfully handled with appropriate imputation strategies

### Statistical Findings:
1. **Smoking Status:** Strongest predictor of insurance charges (correlation: ~0.79)
2. **Age:** Moderate positive correlation with charges (correlation: ~0.30)
3. **BMI:** Weak to moderate correlation with charges (correlation: ~0.20)
4. **Gender & Region:** Minimal impact on charges

### Visualization Insights:
1. **Distribution Analysis:** Charges show right-skewed distribution
2. **Categorical Analysis:** Smokers have significantly higher charges
3. **Correlation Patterns:** Clear linear relationships identified
4. **Interactive Plots:** Enhanced understanding through Plotly visualizations

### Model Performance:
1. **Best Model:** {final_model_name}
2. **Final R² Score:** {final_r2:.4f}
3. **Final RMSE:** ${final_rmse:.2f}
4. **Cross-Validation:** Robust performance across folds

### Technical Skills Demonstrated:
- **Data Preprocessing:** Missing value handling, encoding, type conversion
- **Exploratory Data Analysis:** Comprehensive statistical and visual analysis
- **Machine Learning:** Multiple algorithms with hyperparameter tuning
- **Model Evaluation:** Cross-validation, multiple metrics, residual analysis
- **Visualization:** Static (Matplotlib/Seaborn) and interactive (Plotly) plots
- **Advanced Techniques:** Pipelines, GridSearchCV, progress tracking

### Business Insights:
1. **Risk Assessment:** Smoking status is the primary risk factor
2. **Pricing Strategy:** Age-based pricing should consider smoking interaction
3. **Health Programs:** Focus on smoking cessation programs
4. **Market Segmentation:** Consider demographic-based insurance products

### Next Steps:
1. **Feature Engineering:** Create interaction terms, polynomial features
2. **Advanced Models:** Try ensemble methods, neural networks
3. **Time Series:** Analyze temporal patterns if date data available
4. **External Data:** Incorporate additional health/demographic variables
"""

# %%
"""
## PART 12: ADVANCED TECHNIQUES AND PYTHON METHODS

### Objectives:
- Demonstrate advanced statistical methods (ANOVA, chi-square testing)
- Apply numpy polynomial fitting and mathematical operations
- Implement advanced pandas operations and data manipulation
- Show confidence interval calculations and hypothesis testing
- Demonstrate memory optimization and production-ready techniques
"""

# %%
# ADVANCED TECHNIQUES AND PYTHON METHODS
print("ADVANCED TECHNIQUES AND PYTHON METHODS")
print("="*60)

# 1. ANOVA (f_oneway) for categorical group analysis
print("\n1. ANOVA Testing (f_oneway):")
print("Testing if charges differ significantly across regions")

# Group charges by region for ANOVA
region_groups = [group['charges'].values for name, group in df.groupby('region')]
f_stat, p_value_anova = f_oneway(*region_groups)

print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value_anova:.6f}")
print(f"  Interpretation: {'Significant' if p_value_anova < 0.05 else 'Not significant'} difference between regions")

# 2. Polynomial fitting with np.polyfit and np.poly1d
print("\n2. Polynomial Fitting (np.polyfit & np.poly1d):")

# Fit polynomial to age vs charges relationship
x_data = df['age'].values
y_data = df['charges'].values

degrees = [1, 2, 3]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    # Fit polynomial
    coefficients = np.polyfit(x_data, y_data, degree)
    poly_function = np.poly1d(coefficients)
    
    # Generate smooth curve for plotting
    x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
    y_smooth = poly_function(x_smooth)
    
    # Calculate R²
    y_pred_poly = poly_function(x_data)
    r2_poly = r2_score(y_data, y_pred_poly)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(x_data, y_data, alpha=0.5, s=30)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Degree {degree}')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title(f'Polynomial Degree {degree}\nR² = {r2_poly:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"  Degree {degree}: R² = {r2_poly:.4f}")
    print(f"  Coefficients: {coefficients}")

plt.tight_layout()
plt.show()

# 3. Advanced np.where() usage
print("\n3. Advanced np.where() Conditional Operations:")

# Create risk categories using np.where
df['risk_category'] = np.where(
    df['smoker'] == 'yes',
    np.where(df['age'] > 50, 'High Risk', 'Medium Risk'),
    np.where(df['age'] > 60, 'Medium Risk', 'Low Risk')
)

risk_summary = df.groupby('risk_category')['charges'].agg(['count', 'mean', 'std']).round(2)
print("Risk Categories Analysis:")
print(risk_summary)

# 4. One-hot encoding with pd.get_dummies()
print("\n4. One-Hot Encoding with pd.get_dummies():")

# Create dummy variables for categorical columns
df_encoded = pd.get_dummies(df, columns=['gender', 'smoker', 'region'], prefix=['sex', 'smoke', 'area'])

# Show encoding results
categorical_cols = [col for col in df_encoded.columns if any(prefix in col for prefix in ['sex_', 'smoke_', 'area_'])]
print(f"Original categorical columns → {len(categorical_cols)} binary columns")
print("Encoded columns:", categorical_cols[:5], "..." if len(categorical_cols) > 5 else "")

# Use regex for column selection
import re
smoke_pattern = re.compile(r'^smoke_')
smoke_columns = [col for col in df_encoded.columns if smoke_pattern.match(col)]
print(f"Smoking-related columns: {smoke_columns}")

# 5. Pipeline with multiple preprocessing steps
print("\n5. Advanced Pipeline with Multiple Steps:")

# Create a comprehensive pipeline
comprehensive_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=0.1))
])

# Use numerical features only for this pipeline
X_numerical = df[['age', 'bmi', 'children', 'smoker_encoded', 'gender_encoded']]

# Fit and evaluate pipeline
comprehensive_pipeline.fit(X_numerical, df['charges'])
y_pred_pipeline = comprehensive_pipeline.predict(X_numerical)
r2_comprehensive = r2_score(df['charges'], y_pred_pipeline)

print(f"Comprehensive Pipeline R²: {r2_comprehensive:.4f}")
print("Pipeline steps:")
for i, (name, step) in enumerate(comprehensive_pipeline.steps):
    print(f"  {i+1}. {name}: {type(step).__name__}")

# 6. Cross-validation with cross_val_predict
print("\n6. Advanced Cross-Validation with cross_val_predict:")

# Get cross-validated predictions
cv_predictions = cross_val_predict(comprehensive_pipeline, X_numerical, df['charges'], cv=5)
cv_r2 = r2_score(df['charges'], cv_predictions)

print(f"Cross-validated R²: {cv_r2:.4f}")
print(f"Standard CV R²: {cross_val_score(comprehensive_pipeline, X_numerical, df['charges'], cv=5).mean():.4f}")

# 7. Chi-square test for categorical associations
print("\n7. Chi-Square Test for Categorical Associations:")

# Create contingency table
contingency_table = pd.crosstab(df['smoker'], df['gender'])
print("Contingency Table:")
print(contingency_table)

# Perform chi-square test
chi2_stat, p_val_chi2, dof, expected_freq = chi2_contingency(contingency_table)

print(f"\nChi-square test results:")
print(f"  Chi-square statistic: {chi2_stat:.4f}")
print(f"  P-value: {p_val_chi2:.6f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Expected frequencies:\n{expected_freq}")

# 8. Advanced data manipulation techniques
print("\n8. Advanced Data Manipulation Techniques:")

# Using argsort for custom sorting
age_sorted_indices = df['age'].argsort() # argsort returns indices that would sort the array
print(f"Youngest person's charges: ${df.loc[age_sorted_indices.iloc[0], 'charges']:.2f}")
print(f"Oldest person's charges: ${df.loc[age_sorted_indices.iloc[-1], 'charges']:.2f}")

# Using value_counts() with normalize
print("\nSmoking distribution:")
smoking_dist = df['smoker'].value_counts(normalize=True) * 100
for category, percentage in smoking_dist.items():
    print(f"  {category}: {percentage:.1f}%")

# Mode for most frequent value
most_common_age = df['age'].mode()[0]
print(f"Most common age: {most_common_age}")

# 9. Confidence intervals calculation
print("\n9. Confidence Intervals Calculation:")

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for mean"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of mean
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_err
    return mean - margin_error, mean + margin_error

# Calculate CI for charges by smoking status
for smoker_status in df['smoker'].unique():
    subset_charges = df[df['smoker'] == smoker_status]['charges']
    ci_lower, ci_upper = calculate_confidence_interval(subset_charges)
    mean_charges = subset_charges.mean()
    
    print(f"Smoker {smoker_status}:")
    print(f"  Mean charges: ${mean_charges:.2f}")
    print(f"  95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# 10. Memory optimization techniques
print("\n10. Memory Optimization Techniques:")

# Check memory usage
memory_usage = df.memory_usage(deep=True)
print("Memory usage by column:")
for col, usage in memory_usage.items():
    print(f"  {col}: {usage / 1024:.2f} KB")

# Optimize data types
df_optimized = df.copy()
df_optimized['age'] = df_optimized['age'].astype('int8')  # Age rarely exceeds 127
df_optimized['children'] = df_optimized['children'].astype('int8')

print(f"Original size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"Optimized size: {df_optimized.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\nAdvanced Techniques Section Complete!")
print("All advanced Python and data science methods demonstrated!")

# %%
print("COMPREHENSIVE EDA AND MACHINE LEARNING ANALYSIS COMPLETE!")
print("="*60)
print("All sections successfully executed with detailed insights!")
print("\nTECHNIQUES COVERED:")
print("=" * 50)
techniques = [
    "✓ Data Loading & Preprocessing",
    "✓ Descriptive Statistics & Data Types",
    "✓ 12 Types of Visualizations (Matplotlib & Seaborn)",
    "✓ Correlation Analysis with Statistical Testing", 
    "✓ Interactive Plotly Visualizations",
    "✓ 5 Machine Learning Models",
    "✓ Hyperparameter Tuning with GridSearchCV",
    "✓ Cross-Validation & Model Selection",
    "✓ Feature Importance & Residual Analysis",
    "✓ ANOVA Testing (f_oneway)",
    "✓ Polynomial Fitting (np.polyfit, np.poly1d)",
    "✓ Advanced np.where() Operations",
    "✓ Pipeline Workflows",
    "✓ Chi-Square Testing",
    "✓ Confidence Intervals",
    "✓ One-Hot Encoding",
    "✓ Progress Tracking with TQDM",
    "✓ Memory Optimization",
    "✓ Advanced Pandas Operations"
]

for technique in techniques:
    print(technique)

print("\nREADY FOR GITHUB DEPLOYMENT!")
print("Complete data science workflow from exploration to deployment!")
print("Educational resource for data science learning and practice!") 