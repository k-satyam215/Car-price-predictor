import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the data and the trained model
data = pd.read_csv('Cleaned Car.csv')
req = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

st.title('Welcome to Car Price Predictor')
st.write("This app predicts the price of a car you want to sell. Try filling the details below:")

# Get unique values for select boxes
companies = sorted(data['company'].unique())
car_models = sorted(data['name'].unique()) # Correct variable name to avoid conflict
year = sorted(data['year'].unique(), reverse=True)
fuel_type = data['fuel_type'].unique()

# Create input widgets for user selection
company = st.selectbox("Select the company:", companies)
# The car model list should be filtered based on the selected company
def update_car_model():
    filtered_models = data[data['company'] == company]['name'].unique()
    return sorted(filtered_models)

# Create a filtered list of models
filtered_models = data[data['company'] == company]['name'].unique()
model = st.selectbox("Select the model:", sorted(filtered_models))

year = st.selectbox("Select Year of Purchase:", year)
fuel_type = st.selectbox("Select the Fuel Type:", fuel_type)
kms_driven = st.number_input("Enter the Number of Kilometres that the car has travelled:", min_value=0)

# Predict button
if st.button("Predict Price"):
    try:
        # Create a DataFrame from the user inputs, matching the model's training data structure
        input_data = pd.DataFrame([[model, company, year, kms_driven, fuel_type]], 
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Make the prediction
        prediction = req.predict(input_data)[0]
        
        # Display the prediction to the user
        st.success(f"The predicted price of the car is: ₹ {prediction:,.2f}")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all fields are selected correctly and the model can handle the inputs.")