# -------------------------------
# app.py
# Breast Cancer Prediction Web GUI
# -------------------------------

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# 1️⃣ Paths to model and scaler
MODEL_PATH = os.path.join('model', 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

# 2️⃣ Load model and scaler safely
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
else:
    raise FileNotFoundError(
        "Model or scaler not found in '/model/'. "
        "Please run 'model_building.ipynb' first to generate them."
    )

# 3️⃣ Home route
@app.route('/')
def home():
    return render_template('index.html')

# 4️⃣ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read features from form
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(final_features)[0]
        result = 'Benign' if prediction == 1 else 'Malignant'

        return render_template('index.html', prediction_text=f'Tumor is {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# 5️⃣ Run the app
if __name__ == "__main__":
    app.run(debug=True)
