import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Smart Crop System", layout="centered")

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: #00FF7F;'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

st.write("### Enter soil and weather conditions to get the best crop recommendation")

# Sidebar for inputs
st.sidebar.header("🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0.0)
P = st.sidebar.number_input("Phosphorus (P)", 0.0)
K = st.sidebar.number_input("Potassium (K)", 0.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0)
ph = st.sidebar.number_input("Soil pH", 0.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0)

st.write("---")

# Predict button
if st.button("🌾 Recommend Crop"):

    if N == 0 or P == 0 or K == 0:
        st.warning("⚠️ Please enter valid soil nutrient values.")
    else:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        confidence = np.max(probabilities) * 100

        # Main result
        st.success(f"✅ Recommended Crop: {prediction[0]}")
        st.info(f"📊 Confidence Level: {confidence:.2f}%")

        # Top 3 crops
        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        st.write("### 🌾 Top 3 Recommended Crops:")
        for i in top_indices:
            st.write(f"{model.classes_[i]}: {probs[i]*100:.2f}%")

        # Explanation
        st.write("### 💡 Why this crop?")
        if prediction[0] == "rice":
            st.write("Rice thrives in high rainfall and high humidity conditions.")
        elif prediction[0] == "maize":
            st.write("Maize grows well in moderate rainfall and balanced soil nutrients.")
        elif prediction[0] == "coffee":
            st.write("Coffee requires moderate rainfall and specific temperature ranges.")
        else:
            st.write("This crop matches the given soil nutrients and environmental conditions.")

        # Bar chart visualization
        st.write("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })
        st.bar_chart(prob_df.set_index("Crop"))
# About section
st.write("---")
st.write("### 📘 About This Project")
st.write("This AI-powered system recommends the best crops based on soil nutrients and weather conditions using machine learning models. It helps farmers make data-driven decisions for improved agricultural productivity.")
import streamlit as st

st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #00FF7F;
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #00FF7F;'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>AI-powered system to recommend the best crops based on soil and weather conditions</p>", unsafe_allow_html=True)

st.write("---")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0)
    temperature = st.number_input("Temperature (°C)", 0.0)

with col2:
    P = st.number_input("Phosphorus (P)", 0.0)
    humidity = st.number_input("Humidity (%)", 0.0)

with col3:
    K = st.number_input("Potassium (K)", 0.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0)

ph = st.number_input("Soil pH", 0.0)
if st.button("🌾 Recommend Crop"):
    st.success("Prediction completed!")
st.markdown("### 🌾 Prediction Result")

st.success(f"Recommended Crop: **{prediction[0]}**")

st.info(f"Confidence: {confidence:.2f}%")
st.write("---")

st.markdown("## 📘 About This Project")
st.markdown("""
This AI-powered system helps farmers choose the best crop using:
- Soil nutrients (N, P, K)
- Weather conditions
- Machine Learning models

It improves agricultural productivity and supports data-driven farming.
""")
