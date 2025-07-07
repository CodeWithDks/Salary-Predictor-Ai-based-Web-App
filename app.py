from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Define custom transformers directly in app.py
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

app = Flask(__name__)

# Load the trained model
model_path = 'model/salary_predictor_corrected.pkl'
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def make_prediction(input_data):
    """Make salary prediction using the trained model"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Round to 2 decimal places
        predicted_salary = round(prediction[0], 2)
        
        return predicted_salary, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Education Level': request.form['education'],
            'Years of Experience': int(request.form['experience']),
            'Job Title': request.form['job_title']
        }
        
        # Make prediction
        prediction, error = make_prediction(data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Make prediction
        prediction, error = make_prediction(data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)