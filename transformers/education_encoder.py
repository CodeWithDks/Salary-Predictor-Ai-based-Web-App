# education_encoder.py
from sklearn.base import BaseEstimator, TransformerMixin

class EducationEncoder(BaseEstimator, TransformerMixin):
    """Encodes education level with proper handling"""
    def __init__(self):
        self.education_map = {"Bachelor's": 1, "Master's": 2, "PhD": 3}
        self.unknown_value = 1  # Default to Bachelor's for unknown values
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Education_Level_Encoded'] = X['Education Level'].map(self.education_map).fillna(self.unknown_value)
        return X