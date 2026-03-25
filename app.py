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