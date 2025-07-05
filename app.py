from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model, scaler, and label encoder
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature names (must match training)
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    try:
        input_values = [float(request.form[feature]) for feature in feature_names]
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        predicted_class = label_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        return f"❌ Error: {e}"

    return render_template('index.html', prediction_text=f'Predicted Wine Quality: {predicted_class}', feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)