import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('wine_quality_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Wine Quality Prediction 🍷')

st.write("Enter the wine characteristics below:")

# Input fields for the features
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, format="%.2f")
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, format="%.2f")
citric_acid = st.number_input('Citric Acid', min_value=0.0, format="%.2f")
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, format="%.2f")
chlorides = st.number_input('Chlorides', min_value=0.0, format="%.5f")
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, format="%.2f")
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, format="%.2f")
density = st.number_input('Density', min_value=0.0, format="%.5f")
pH = st.number_input('pH', min_value=0.0, format="%.2f")
sulphates = st.number_input('Sulphates', min_value=0.0, format="%.2f")
alcohol = st.number_input('Alcohol', min_value=0.0, format="%.2f")

# When the user clicks the Predict button
if st.button('Predict Wine Quality'):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                           chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                           density, pH, sulphates, alcohol]])
    
    prediction = model.predict(features)
    st.success(f'The predicted wine quality is: {prediction[0]}')

