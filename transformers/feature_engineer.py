# feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates new features from existing ones"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure numeric columns are properly typed
        X['Age'] = pd.to_numeric(X['Age'], errors='coerce')
        X['Years of Experience'] = pd.to_numeric(X['Years of Experience'], errors='coerce')
        
        # Basic features
        X['Experience_Squared'] = X['Years of Experience'] ** 2
        X['Age_Experience_Ratio'] = X['Age'] / (X['Years of Experience'] + 1)

        # Career stage based on experience
        X['Career_Stage'] = np.where(
            X['Years of Experience'] < 5, 'Early',
            np.where(X['Years of Experience'] < 15, 'Mid', 'Late')
        )

        # Age buckets
        X['Age_Group'] = pd.cut(X['Age'], bins=[0, 30, 40, 50, 100],
                               labels=['Under_30', '30-40', '40-50', 'Over_50'])
        X['Age_Group'] = X['Age_Group'].astype(str)
        
        return X