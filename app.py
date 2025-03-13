from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the regression model and scaler
regressor = pickle.load(open("regressor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(request.form[key]) for key in ['length1', 'length2', 'length3', 'height', 'width']]
        
        # Preprocess input
        features_scaled = scaler.transform([features])
        
        # Predict using regression model
        prediction = regressor.predict(features_scaled)[0]
        
        return render_template('index.html', prediction_text=f'Predicted Weight: {prediction:.2f} g')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
