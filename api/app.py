import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Get the base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'model (6).pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vector.pkl') # Make sure this is your TfidfVectorizer/CountVectorizer

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    if model is None:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if vectorizer is None:
        with open(VECTOR_PATH, 'rb') as f:
            vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()
        
        # Get data from the frontend form
        if request.is_json:
            data = request.json.get('input_data', '')
        else:
            data = request.form.get('input_data', '')

        if not data.strip():
            return jsonify({'error': 'No input text provided'}), 400

        # 1. Transform the raw text into numerical features using your vectorizer
        transformed_data = vectorizer.transform([data])
        
        # 2. Make the prediction using your Logistic Regression model
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)

        # 3. Extract output values safely
        result = str(prediction[0])
        confidence = float(np.max(prediction_proba))

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Required for Vercel
app.debug = False
