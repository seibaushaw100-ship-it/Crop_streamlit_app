import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import altair as alt

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌱",
    layout="wide"
)

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;  /* dark main background */
    color: #E0E0E0;  /* soft text */
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    padding: 20px;
}

/* Headers */
h1, h2, h3 {
    color: #00FF7F;
    text-align: center;
    font-family: 'Arial', sans-serif;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00FF7F, #00c853);
    color: black;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #00c853, #00FF7F);
}

/* Card container for prediction */
.card {
    background: linear-gradient(135deg, #1E1E1E, #2C2C2C);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

/* Metrics text */
.stMetric label, .stMetric div {
    color: #E0E0E0;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Model & Scaler
# ------------------------------
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.success("✅ Model Ready")
st.sidebar.markdown("## 🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0.0)
P = st.sidebar.number_input("Phosphorus (P)", 0.0)
K = st.sidebar.number_input("Potassium (K)", 0.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0)
ph = st.sidebar.number_input("Soil pH", 0.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0)

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered crop prediction for smart farming decisions</p>", unsafe_allow_html=True)
st.write("---")

# ------------------------------
# Layout Columns
# ------------------------------
col1, col2 = st.columns([1, 2])

with col2:
    st.subheader("🌾 Prediction Result")

    if st.button("🚀 Recommend Crop"):

        # Input validation
        if N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid soil nutrient values.")
        else:
            # Prepare data
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            confidence = np.max(probabilities) * 100

            # ------------------------------
            # Prediction Card
            # ------------------------------
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#00FF7F;">🌾 Recommended Crop</h2>
                <h1 style="color:white; font-size:40px;">{prediction[0].upper()}</h1>
            </div>
            """, unsafe_allow_html=True)

            # ------------------------------
            # Metrics
            # ------------------------------
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{confidence:.2f}%")
            m2.metric("Soil pH", f"{ph}")

            # ------------------------------
            # Top 3 Crops
            # ------------------------------
            st.write("### 🌾 Top 3 Crops")
            probs = probabilities[0]
            top_indices = probs.argsort()[-3:][::-1]

            for i in top_indices:
                st.progress(float(probs[i]))
                st.write(f"🌱 {model.classes_[i]} → {probs[i]*100:.2f}%")

            # ------------------------------
            # Why this crop?
            # ------------------------------
            st.write("### 💡 Why this crop?")
            if prediction[0] == "rice":
                st.write("Rice thrives in high rainfall and humidity.")
            elif prediction[0] == "maize":
                st.write("Maize grows well in moderate rainfall and balanced nutrients.")
            elif prediction[0] == "coffee":
                st.write("Coffee requires specific temperature and rainfall conditions.")
            else:
                st.write("This crop matches your soil and weather conditions.")

            # ------------------------------
            # Prediction Distribution Chart
            # ------------------------------
            prob_df = pd.DataFrame({
                "Crop": model.classes_,
                "Probability": probabilities[0]
            })

            chart = alt.Chart(prob_df).mark_bar().encode(
                x='Crop',
                y='Probability',
                color=alt.Color('Crop', scale=alt.Scale(range=['#00FF7F', '#00c853', '#43A047', '#E53935']))
            ).properties(width=600, height=300)

            st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.markdown("""
<center>
🌍 Made with ❤️ for Smart Agriculture | AI Crop Recommendation System
</center>
""", unsafe_allow_html=True)