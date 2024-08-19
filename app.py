import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the saved scaler and model
scaler = joblib.load("scaler.pkl")
model = load_model("model.h5")

# Streamlit app interface
st.title('Sternal Notch Prediction')

# User input for height in inches
height_inch = st.number_input('Enter your height in inches:', min_value=0.0, step=0.1)

# When the user clicks the 'Predict' button
if st.button('Predict Sternal Notch'):
    if height_inch > 0:
        # Scale the user input using the loaded scaler
        height_inch_scaled = scaler.transform(np.array([[height_inch]]))
        
        # Predict the sternal notch using the loaded model
        prediction = model.predict(height_inch_scaled)
        
        # Display the prediction
        st.write(f'The predicted sternal notch is {prediction[0][0]:.2f} cm')
    else:
        st.write("Please enter a valid height in inches.")
