from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os
import datetime
import threading
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory and CSV file
DATA_DIR = Path('data')
DATA_FILE = DATA_DIR / 'predictions.csv'
DATA_DIR.mkdir(exist_ok=True)

# Initialize CSV with headers if it doesn’t exist
if not DATA_FILE.exists():
    pd.DataFrame(columns=[
        'Timestamp', 'Glucose', 'BMI', 'Pregnancies', 'Age', 'BloodPressure',
        'DiabetesPedigreeFunction', 'Result', 'Probability'
    ]).to_csv(DATA_FILE, index=False)

# Thread lock for safe CSV writes
csv_lock = threading.Lock()

# Load model files
try:
    model = joblib.load('models/diabetes_model_final.pkl')
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    logger.info("Model files loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model files: {e}")
    raise Exception("Model files not found or corrupted")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data with validation
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        pregnancies = float(request.form['pregnancies'])
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])

        # Validate ranges
        if not (0 < glucose <= 200):
            return render_template('index.html', error="Glucose must be between 0 and 200 mg/dL")
        if not (10 <= bmi <= 60):
            return render_template('index.html', error="BMI must be between 10 and 60 kg/m²")
        if not (0 <= pregnancies <= 20):
            return render_template('index.html', error="Pregnancies must be between 0 and 20")
        if not (15 <= age <= 100):
            return render_template('index.html', error="Age must be between 15 and 100 years")
        if not (0 <= blood_pressure <= 200):
            return render_template('index.html', error="Blood Pressure must be between 0 and 200 mmHg")
        if not (0.1 <= diabetes_pedigree <= 2.5):
            return render_template('index.html', error="Diabetes Pedigree Function must be between 0.1 and 2.5")

        # Create patient DataFrame
        patient_data = pd.DataFrame({
            'Glucose': [glucose],
            'BMI': [bmi],
            'Pregnancies': [pregnancies],
            'Age': [age],
            'BloodPressure': [blood_pressure],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Glucose_BMI': [glucose * bmi]
        })

        # Impute missing values
        zero_cols = ['Glucose', 'BloodPressure', 'BMI']
        patient_data[zero_cols] = imputer.transform(patient_data[zero_cols])

        # Scale features
        patient_data_scaled = scaler.transform(patient_data)

        # Predict
        probability = model.predict_proba(patient_data_scaled)[:, 1][0]
        prediction = 1 if probability >= 0.5 else 0
        result = 'Diabetes Risk Detected' if prediction == 1 else 'No Diabetes Risk'
        probability_percent = round(probability * 100, 2)

        # Save data to CSV
        with csv_lock:
            data_entry = pd.DataFrame([{
                'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Glucose': glucose,
                'BMI': bmi,
                'Pregnancies': pregnancies,
                'Age': age,
                'BloodPressure': blood_pressure,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Result': result,
                'Probability': probability_percent
            }])
            data_entry.to_csv(DATA_FILE, mode='a', header=False, index=False)

        logger.info(f"Prediction made: {result}, Probability: {probability_percent}%")
        return render_template('result.html', result=result, probability=probability_percent)

    except ValueError:
        logger.warning("Invalid input received")
        return render_template('index.html', error="Please enter valid numeric values")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html', error="An error occurred. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))