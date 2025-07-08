from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class JobGrouper(BaseEstimator, TransformerMixin):
    """Groups similar job titles into categories"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def group_job_titles(title):
            if pd.isna(title):
                return "Other"
            title = str(title).lower()
            if any(word in title for word in ['manager', 'director', 'vp', 'chief', 'executive']):
                return "Leadership"
            elif any(word in title for word in ['engineer', 'developer', 'scientist', 'technical']):
                return "Technical"
            elif any(word in title for word in ['analyst', 'research', 'data']):
                return "Analyst"
            elif any(word in title for word in ['sales', 'account', 'business development']):
                return "Sales"
            elif any(word in title for word in ['marketing', 'brand', 'product']):
                return "Marketing"
            elif any(word in title for word in ['hr', 'human resource', 'recruiter']):
                return "HR"
            else:
                return "Other"

        X['Job_Group'] = X['Job Title'].apply(group_job_titles)
        return X
