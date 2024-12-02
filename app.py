# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'rf_model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    message = request.form['mixedInput']   # Get the message input from the form

    # Transform the input message using the vectorizer
    message_vectorized = vectorizer.transform([message])  # Ensure this matches your model's input requirements
    
    # Make prediction
    prediction = model.predict(message_vectorized)
    output = 'Spam' if prediction[0] == 1 else 'Not Spam'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output), message=message)  #message=message was added to keep the entered message in the input field after the form is submitted, you can pre-fill the input field with the previous user input. You can achieve this by passing the message back to the template and using it as the value for the input field.

if __name__ == "__main__":
    app.run(debug=True)