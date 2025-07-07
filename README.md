# Salary Predictor Web Application

A modern, AI-powered salary prediction web application built with Flask and Machine Learning. This application uses a Random Forest Regressor model to predict salaries based on various factors including age, gender, education level, years of experience, and job title.

## Features

- **Machine Learning Powered**: Uses Random Forest algorithm with advanced feature engineering
- **Modern UI**: Responsive design with glassmorphism effects and smooth animations
- **Real-time Validation**: Form validation with instant feedback
- **Interactive Results**: Animated salary display with detailed breakdown
- **Mobile Responsive**: Fully responsive design for all devices
- **Developer Info**: Credits section with social media links

## Project Structure

```
salary-predictor/
├── app.py                          # Main Flask application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── Data/
│   └── Salary Data.csv            # Training dataset
├── model/
│   └── salary_predictor_corrected.pkl  # Trained model
├── templates/
│   └── index.html                 # Main HTML template
└── static/
    ├── style.css                  # CSS styles
    └── script.js                  # JavaScript functionality
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd salary-predictor
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

Make sure you have the dataset in the `Data/` folder and run:

```bash
python train_model.py
```

This will:
- Process the data
- Train the Random Forest model
- Save the trained model to `model/salary_predictor_corrected.pkl`
- Display model performance metrics

### Step 5: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

### Web Interface
- `GET /` - Main web interface

### API Endpoints
- `POST /predict` - Form-based prediction
- `POST /api/predict` - JSON API for predictions
- `GET /health` - Health check endpoint

### API Usage Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 32,
    "Gender": "Female",
    "Education Level": "Master'\''s",
    "Years of Experience": 7,
    "Job Title": "Software Engineer"
  }'
```

## Model Details

### Algorithm
- **Random Forest Regressor** with hyperparameter tuning
- **Cross-validation** for robust performance evaluation
- **Feature engineering** for improved predictions

### Features Used
- Age
- Gender
- Education Level (Bachelor's, Master's, PhD)
- Years of Experience
- Job Title (grouped into categories)
- Derived features (Experience², Age/Experience ratio, Career stage)

### Performance Metrics
- R² Score optimization
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Residual analysis for model validation

## Key Features

### Data Processing
- **Salary Correction**: Handles missing zeros in salary data
- **Job Grouping**: Categorizes job titles into meaningful groups
- **Feature Engineering**: Creates additional predictive features
- **Education Encoding**: Proper handling of education levels

### Model Pipeline
- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical
- **Custom Transformers**: SalaryCorrector, JobGrouper, FeatureEngineer
- **Pipeline Integration**: Seamless data flow from input to prediction

### Web Application
- **Form Validation**: Client-side and server-side validation
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Loading States**: Visual feedback during prediction
- **Responsive Design**: Mobile-first approach

## Customization

### Updating the Model
1. Modify `train_model.py` with your changes
2. Retrain the model: `python train_model.py`
3. Restart the Flask application

### Styling
- Modify `static/style.css` for visual changes
- Update `static/script.js` for interactive features
- Edit `templates/index.html` for layout changes

### Developer Information
Update the developer section in `templates/index.html`:
- Change social media links
- Update developer name
- Modify portfolio links

## Deployment

### Local Development
```bash
python app.py
```

### Production (using Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Developer

**Deepak Singh**
- LinkedIn: [linkedin.com/in/deepak-singh](https://linkedin.com/in/deepak-singh)
- GitHub: [github.com/deepak-singh](https://github.com/deepak-singh)
- Portfolio: [deepak-singh.portfolio.com](https://deepak-singh.portfolio.com)

## Acknowledgments

- Built with Flask and scikit-learn
- UI inspired by modern web design trends
- Machine learning pipeline optimized for production use

---

For questions or support, please open an issue or contact the developer through the provided social media links.