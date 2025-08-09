from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import sys
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "kidney_new.pkl"
MODEL_PORTABLE = "kidney_new_portable.joblib"

# Try loading the portable version first
model = None
try:
    if os.path.exists(MODEL_PORTABLE):
        model = joblib.load(MODEL_PORTABLE)
        print("Loaded portable model successfully.")
    else:
        try:
            # Try loading the original pickle
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("Loaded original pickle model successfully.")

            # Save as portable version
            joblib.dump(model, MODEL_PORTABLE)
            print(f"Converted model saved as {MODEL_PORTABLE}")
        except Exception as e:
            print(f"Error loading pickle model: {e}", file=sys.stderr)
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found.", file=sys.stderr)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/indexnew")
def prediction_page():
    return render_template('indexnew.html')

@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded. Please check server logs.", 500

    try:
        # Determine feature names
        if hasattr(model, 'feature_names_in_'):
            features_name = list(model.feature_names_in_)
        else:
            features_name = [
                'white blood cell count', 'blood urea', 'blood glucose random',
                'serum creatinine', 'packed cell volume', 'albumin',
                'haemoglobin', 'age', 'sugar', 'hypertension'
            ]

        print("\n=== RAW FORM DATA ===")
        print(request.form.to_dict())

        # Convert inputs
        input_features = []
        for name in features_name:
            val = request.form.get(name)
            if val is None or val.strip() == "":
                raise ValueError(f"Missing value for: {name}")
            input_features.append(float(val))

        df = pd.DataFrame([input_features], columns=features_name)
        print("\n=== PROCESSED DATAFRAME ===")
        print(df)

        # Prediction
        output = model.predict(df)
        pred_class = int(output[0])  # Now explicitly cast to int

        if hasattr(model, 'predict_proba'):
            pred_probs = model.predict_proba(df)[0]
            print("\nPrediction probabilities:", pred_probs)
        else:
            pred_probs = None
            print("\nModel does not support probability output.")

        output_label = {0: "NO CKD", 1: "CKD"}
        prediction_text = output_label.get(pred_class, "Unknown")

        print(f"Predicted class: {pred_class} ({prediction_text})")
        print("=========================\n")

        return render_template('result.html', prediction_text=prediction_text, probabilities=pred_probs)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('home.html', error_message=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
