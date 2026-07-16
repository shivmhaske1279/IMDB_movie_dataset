import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Get the base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'model (6).pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vector.pkl')

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
        
        # Get data from form or JSON
        if request.is_json:
            data = request.json.get('input_data', '')
        else:
            data = request.form.get('input_data', '')

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Adjust this processing step based on what your vector.pkl does.
        # If vector.pkl is actually a TfidfVectorizer/CountVectorizer:
        # transformed_data = vectorizer.transform([data])
        
        # If vector.pkl is another model or expects an array:
        # For demonstration, assuming text/feature transformation or direct array passing:
        if hasattr(vectorizer, 'transform'):
            transformed_data = vectorizer.transform([data])
        else:
            # Fallback if it requires numeric array input directly into the model
            transformed_data = np.array([float(x) for x in data.split(',')]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)

        # Get the predicted class label
        result = str(prediction[0])
        confidence = float(np.max(prediction_proba)[0]) if hasattr(prediction_proba, 'ndim') and prediction_proba.ndim > 1 else float(np.max(prediction_proba))

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Required for Vercel
app.debug = False
