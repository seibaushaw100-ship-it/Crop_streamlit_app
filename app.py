import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

#  PAGE CONFIG
st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌱",
    layout="wide"
)

# UI STYLING
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #f0fdf4);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}

/* Only style text, NOT everything */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Fix number input controls */
[data-testid="stSidebar"] button {
    background-color: #d1fae5 !important;
    color: black !important;
    border-radius: 5px;
}
/* REMOVED INPUT OVERRIDE THAT BROKE + / - */

.glass {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    text-align: center;
}

.stButton>button {
    background: linear-gradient(90deg, #22c55e, #15803d);
    color: white;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
}

h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR INPUTS (WITH LIMITS)
st.sidebar.header(" 🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, step=1.0)
P = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, step=1.0)
K = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, step=1.0)

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, step=0.5)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, step=1.0)

ph = st.sidebar.number_input("Soil pH", 0.0, 14.0, step=0.1)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 300.0, step=1.0)

# LOAD MODEL
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
    return model, scaler

model, scaler = load_model()

# TITLE
st.markdown("<h1> 🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["📘 About", "🌾 Prediction", "📊 Insights"])

# 📘 ABOUT + HOW TO USE
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📘 About This System</h3>
    <p>This system helps farmers choose the best crop based on soil and weather conditions using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3> 🛠️  How to Use This App</h3>
    <ol style="text-align:left; font-size:16px;">
        <li>Go to the <b> 🌾 Prediction</b> tab</li>
        <li>Enter soil nutrients (N, P, K)</li>
        <li>Enter weather data (Temperature, Humidity, Rainfall)</li>
        <li>Enter soil pH</li>
        <li>Click <b>Recommend Crop</b></li>
        <li>View results and confidence level</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.info("Tip:💡 Use realistic values for best results.")

    st.markdown('</div>', unsafe_allow_html=True)

#  PREDICTION
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if st.button(" 🌱 Recommend Crop", use_container_width=True):

        if any(v == 0 for v in [N, P, K]):
            st.warning("⚠️ Please enter valid soil nutrient values.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_data = scaler.transform(input_data)

            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]

            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities

    # SHOW RESULT
    if "prediction" in st.session_state:

        crop = st.session_state["prediction"]
        probas = st.session_state["probabilities"]
        classes = model.classes_

        confidence = np.max(probas) * 100

        # Result Card (FIXED VISIBILITY)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #22c55e, #15803d);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;">
            🌾 {crop.upper()}
        </div>
        """, unsafe_allow_html=True)

        # 📊 Confidence
        st.markdown("### 📊 Confidence Level")

        if confidence > 75:
            st.success(f"High Confidence: {confidence:.2f}%")
        elif confidence > 50:
            st.warning(f"Moderate Confidence: {confidence:.2f}%")
        else:
            st.error(f"Low Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        st.divider()

        #  Top 3
        st.subheader("🔝 Top 3 Predictions")
        top3_idx = np.argsort(probas)[-3:][::-1]

        for i in top3_idx:
            st.write(f" 🌱 **{classes[i]}** — {probas[i]*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

#  INSIGHTS
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if "prediction" in st.session_state:

        probabilities = st.session_state["probabilities"]

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Crop"))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 💡 Why this crop?")

        crop_explanations = {
            "rice": "Rice thrives in high rainfall and humidity.",
            "maize": "Maize grows well in balanced nutrients.",
            "coffee": "Coffee requires stable temperature and rainfall.",
        }

        crop_name = st.session_state["prediction"].lower()

        st.write(crop_explanations.get(
            crop_name,
            "This crop matches the given soil nutrients and environmental conditions."
        ))

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run a prediction first.")

    st.markdown('</div>', unsafe_allow_html=True)

#  FOOTER
st.write("---")
st.markdown("""
<center>
🌍 Made with ❤️ for Smart Agriculture | AI Crop Recommendation System
</center>
""", unsafe_allow_html=True)