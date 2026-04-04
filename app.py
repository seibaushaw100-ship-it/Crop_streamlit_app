import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# =========================
# ⚙️ PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌱",
    layout="wide"
)

# =========================
# 🎨 UI STYLING
# =========================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #f0fdf4);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] input {
    color: black !important;
    background-color: #ecfdf5 !important;
}

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

# =========================
# 📥 SIDEBAR INPUTS
# =========================
st.sidebar.header("🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0)

# =========================
# ⚡ LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
    return model, scaler

model, scaler = load_model()

# =========================
# 🌱 TITLE
# =========================
st.markdown("<h1>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# =========================
# 📑 TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📘 About", "🌾 Prediction", "📊 Insights"])

# =========================
# 📘 ABOUT
# =========================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📘 About This System</h3>
    <p>This system helps farmers choose the best crop based on soil and weather conditions using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🌾 PREDICTION
# =========================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if st.button("🌱 Recommend Crop", use_container_width=True):

        if any(v == 0 for v in [N, P, K]):
            st.warning("⚠️ Please enter valid soil nutrient values.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled_data = scaler.transform(input_data)

            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]

            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities

    # ✅ SHOW RESULT
    if "prediction" in st.session_state:

        crop = st.session_state["prediction"]
        probas = st.session_state["probabilities"]
        classes = model.classes_

        top_idx = np.argmax(probas)
        confidence = probas[top_idx] * 100

        # 🌾 Crop Card
        st.markdown(f"""
        <div class="card">
            <h2>🌾 {crop.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)

        # 📊 Confidence Section
        st.markdown("### 📊 Confidence Level")

        if confidence > 75:
            st.success(f"High Confidence: {confidence:.2f}%")
        elif confidence > 50:
            st.warning(f"Moderate Confidence: {confidence:.2f}%")
        else:
            st.error(f"Low Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        st.divider()

        # 🌾 Top 3 Predictions
        st.subheader("🔝 Top 3 Predictions")
        top3_idx = np.argsort(probas)[-3:][::-1]

        for i in top3_idx:
            st.write(f"🌱 **{classes[i]}** — {probas[i]*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📊 INSIGHTS
# =========================
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
            "maize": "Maize grows well in balanced soil nutrients.",
            "coffee": "Coffee requires stable temperature and rainfall.",
        }

        crop_name = st.session_state["prediction"].lower()

        st.write(crop_explanations.get(
            crop_name,
            "This crop best matches your soil and weather conditions."
        ))

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run a prediction first.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🧾 FOOTER
# =========================
st.write("---")
st.markdown("""
<center>
🌍 Made with ❤️ for Smart Agriculture | AI Crop Recommendation System
</center>
""", unsafe_allow_html=True)