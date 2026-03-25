import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page config (ONLY ONCE at top)
st.set_page_config(page_title="Smart Crop System", page_icon="🌱", layout="centered")

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: #00FF7F;'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

st.write("### Enter soil and weather conditions")

# Inputs (clean layout)
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0)
    P = st.number_input("Phosphorus (P)", 0.0)
    K = st.number_input("Potassium (K)", 0.0)
    ph = st.number_input("Soil pH", 0.0)

with col2:
    temperature = st.number_input("Temperature (°C)", 0.0)
    humidity = st.number_input("Humidity (%)", 0.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0)

st.write("---")

# Button (ONLY ONE)
if st.button("🌾 Recommend Crop"):

    if N == 0 or P == 0 or K == 0:
        st.warning("⚠️ Enter valid values")
    else:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        confidence = np.max(probabilities) * 100

        st.success(f"✅ Recommended Crop: {prediction[0]}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        # Top 3
        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        st.write("### 🌾 Top 3 Crops")
        for i in top_indices:
            st.write(f"{model.classes_[i]}: {probs[i]*100:.2f}%")

        # Chart
        st.write("### 📊 Probabilities")
        df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })
        st.bar_chart(df.set_index("Crop"))

# About
st.write("---")
st.write("### 📘 About This Project")
st.write("AI system that recommends crops based on soil and weather using machine learning.")