from flask import Flask, render_template, request, flash, redirect

import numpy as np
import pandas as pd
import joblib





app = Flask(__name__)

model = joblib.load(open('kidney_new.pkl','rb'))

print(type(model))
print(model.get_params())  # If it's a scikit-learn model

@app.route("/")
def home():
    """
    Renders the home page.
    """
    return render_template('home.html')

@app.route("/indexnew")
def prediction_page():
    """
    Renders the prediction form page for CKD.
    """
    return render_template('indexnew.html')

@app.route("/predict", methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    It collects the form data, validates it, and returns the prediction result in words.
    """
    if model is None:
        return "Error: Model not loaded. Please check the server logs.", 500

    try:
        # A list of the feature names, in the exact order the model expects.
        # This list must match the form field names in indexnew.html.
        # The order has been corrected to match the kidney.pkl model's requirements.
        features_name = ['white blood cell count', 'blood urea', 'blood glucose random',
       'serum creatinine', 'packed cell volume', 'albumin', 'haemoglobin',
       'age', 'sugar', 'hypertension']
        
        # Collect values from the form, ensuring the order matches features_name.
        input_features = [float(request.form[name]) for name in features_name]
        
        # Create a Pandas DataFrame for the model
        df = pd.DataFrame([input_features], columns=features_name)
        
        # Make a prediction using the loaded model.
        output=model.predict(df)
        
        # Extract the prediction result (which is a single value from the numpy array)
        predicted_class = int(output[0])
        
        # Map the numerical output to a user-friendly string.
        output_label = {0: "NO CKD", 1: " CKD"}
        prediction_text = output_label[predicted_class]
        
        # Render the result page with the prediction text.
        return render_template('result.html', prediction_text=prediction_text)
    
    except Exception as e:
        # Catch any exceptions that occur during the prediction process
        print(f"Error during prediction: {e}")
        # If an error occurs, redirect back to the home page with an error message
        return render_template('home.html', error_message=f"An error occurred: {e}. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)

