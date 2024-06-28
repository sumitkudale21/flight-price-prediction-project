import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and preprocessor
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# User input
st.title("Flight Price Prediction")

Airline = st.selectbox("Airline", ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy'])
Source = st.selectbox("Source", ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
Destination = st.selectbox("Destination", ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])
Total_Stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
Date = st.number_input("Date", min_value=1, max_value=31, value=1)
Month = st.number_input("Month", min_value=1, max_value=12, value=1)
Dep_time = st.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening', 'Night'])
Arrival_time = st.selectbox("Arrival Time", ['Morning', 'Afternoon', 'Evening', 'Night'])
Duration_time = st.number_input("Duration (in hours)", min_value=0.0, max_value=24.0, value=1.0)

# Create a dictionary with the user input
user_input = {
    'Airline': Airline,
    'Source': Source,
    'Destination': Destination,
    'Total_Stops': Total_Stops,
    'Date': Date,
    'Month': Month,
    'Dep_time': Dep_time,
    'Arrival_time': Arrival_time,
    'Duration_time': Duration_time
}

# Convert the user input to a DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

def predict_price(user_input):
    # Preprocess the input data using the loaded preprocessor
    user_input_processed = preprocessor.transform(user_input)
    # Make the prediction
    prediction = model.predict(user_input_processed)
    return prediction[0]

# Predict button
if st.button("Predict"):
    prediction_result = predict_price(user_input_df)
    st.write(f"Estimated Flight Price: {prediction_result:.2f}")