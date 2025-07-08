import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler,
                                 FunctionTransformer, PowerTransformer)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
import shap
import joblib
import warnings
from transformers.feature_engineer import FeatureEngineer
from transformers.education_encoder import EducationEncoder
from transformers.job_grouper import JobGrouper
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_SALARY = 1000

class SalaryCorrector(BaseEstimator, TransformerMixin):
    """Corrects salary values that appear to be missing zeros"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Salary' in X.columns:
            X['Salary'] = X['Salary'].apply(lambda x: x if x > THRESHOLD_SALARY else x * 100)
        return X
    

def load_data(filepath):
    """Load and initial process of raw data"""
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    df = df.dropna().copy()
    df = df.drop_duplicates().copy()
    
    print(f"After cleaning shape: {df.shape}")
    
    # Convert dtypes
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Years of Experience'] = pd.to_numeric(df['Years of Experience'], errors='coerce')
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    
    # Remove any rows with conversion errors
    df = df.dropna()
    
    return df

def build_complete_pipeline():
    """Build complete pipeline with all transformations"""
    
    # Define feature sets
    numeric_features = ['Age', 'Years of Experience', 'Experience_Squared', 'Age_Experience_Ratio']
    categorical_features = ['Gender', 'Job_Group', 'Career_Stage', 'Age_Group']
    education_feature = ['Education_Level_Encoded']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
            ('edu', StandardScaler(), education_feature)
        ],
        remainder='drop'
    )
    
    # Complete pipeline
    pipeline = Pipeline([
        #('salary_corrector', SalaryCorrector()),
        ('job_grouper', JobGrouper()),
        ('feature_engineer', FeatureEngineer()),
        ('education_encoder', EducationEncoder()),
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    
    return pipeline

def analyze_data(df):
    """Generate comprehensive EDA"""
    print("=== Basic Statistics ===")
    print(df.describe())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Unique Values ===")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()} unique values")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Salary distribution
    axes[0, 0].hist(df['Salary'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Salary Distribution')
    axes[0, 0].set_xlabel('Salary')
    axes[0, 0].set_ylabel('Frequency')
    
    # Experience vs Salary
    axes[0, 1].scatter(df['Years of Experience'], df['Salary'], alpha=0.6)
    axes[0, 1].set_title('Experience vs Salary')
    axes[0, 1].set_xlabel('Years of Experience')
    axes[0, 1].set_ylabel('Salary')
    
    # Salary by Gender
    if 'Gender' in df.columns:
        sns.boxplot(data=df, x='Gender', y='Salary', ax=axes[1, 0])
        axes[1, 0].set_title('Salary by Gender')
    
    # Age vs Salary
    axes[1, 1].scatter(df['Age'], df['Salary'], alpha=0.6, color='orange')
    axes[1, 1].set_title('Age vs Salary')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Salary')
    
    plt.tight_layout()
    plt.show()

def tune_model(pipeline, X_train, y_train):
    """Hyperparameter tuning"""
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Comprehensive model evaluation"""
    # Training predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n=== Model Performance ===")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print("⚠️  Warning: Potential overfitting detected")
    
    # Residual analysis
    residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_test, y_test_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Salary')
    axes[0].set_ylabel('Predicted Salary')
    axes[0].set_title('Predicted vs Actual')
    
    # Residual plot
    axes[1].scatter(y_test_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Salary')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    # Residual distribution
    axes[2].hist(residuals, bins=20, alpha=0.7, color='lightblue')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return test_r2, test_rmse, test_mae

def make_prediction(model, sample_data):
    """Make prediction with proper data structure"""
    # Ensure the input has the right structure
    if isinstance(sample_data, dict):
        sample_data = pd.DataFrame([sample_data])
    
    try:
        prediction = model.predict(sample_data)
        return prediction[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    """Main execution function"""
    # Load data
    df = load_data('Data/Salary Data.csv')
    
    # Analyze raw data
    analyze_data(df)
    
    # Prepare features and target
    # Remove target from features, but keep Job Title for transformation
    X = df.drop(['Salary'], axis=1)
    y = df['Salary']
    
    # Split data BEFORE any transformations
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Build pipeline
    pipeline = build_complete_pipeline()
    
    # Tune model
    best_model = tune_model(pipeline, X_train, y_train)
    
    # Evaluate model
    r2, rmse, mae = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    
    # Save model
    joblib.dump(best_model, 'model/salary_predictor_corrected.pkl')
    print("Model saved successfully!")
    
    # Example prediction
    sample_data = {
        'Age': 32,
        'Gender': 'Female',
        'Education Level': "Master's",
        'Years of Experience': 7,
        'Job Title': 'Software Engineer'
    }
    
    predicted_salary = make_prediction(best_model, sample_data)
    if predicted_salary:
        print(f"\nExample Prediction:")
        print(f"Input: {sample_data}")
        print(f"Predicted Salary:{predicted_salary:,.2f}")

if __name__ == "__main__":
    main()