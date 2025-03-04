# app.py
import streamlit as st
import pickle
import numpy as np
from googletrans import Translator

# Load the trained model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize translator
translator = Translator()

# Streamlit app
st.title("🏠 House Price Prediction")

# Input fields
st.sidebar.header("Input Features")

MedInc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, max_value=15.0, value=3.0)
HouseAge = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, max_value=100.0, value=30.0)
AveRooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, max_value=20.0, value=5.0)
AveBedrms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, max_value=10.0, value=2.0)
Population = st.sidebar.number_input("Population (Population)", min_value=0.0, max_value=10000.0, value=1000.0)
AveOccup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, max_value=10.0, value=3.0)
Latitude = st.sidebar.number_input("Latitude (Latitude)", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.sidebar.number_input("Longitude (Longitude)", min_value=-125.0, max_value=-114.0, value=-118.0)

# Add polynomial features
AveRoomsSq = AveRooms ** 2
AveBedrmsSq = AveBedrms ** 2
PopulationSq = Population ** 2
HouseAgeSq = HouseAge ** 2
AveOccupSq = AveOccup ** 2

# Language selection
language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French", "German", "Chinese"])

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude,
                           AveRoomsSq, AveBedrmsSq, PopulationSq, HouseAgeSq, AveOccupSq]).reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)

    # Translate output
    if language != "English":
        translated_output = translator.translate(f"Predicted House Price: ${output}", dest=language.lower()).text
    else:
        translated_output = f"Predicted House Price: ${output}"

    # Display result
    st.success(translated_output)

# Display dataset description
st.subheader("Dataset Description")
st.write("""
This app uses the California Housing Dataset to predict house prices.
The dataset contains the following features:
- **MedInc**: Median income in the area
- **HouseAge**: Median age of houses in the area
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Population in the area
- **AveOccup**: Average number of household members
- **Latitude**: Latitude of the area
- **Longitude**: Longitude of the area
""")