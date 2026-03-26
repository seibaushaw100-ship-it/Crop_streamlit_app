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
    background: linear-gradient(135deg, #dbeafe, #f0fdf4) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}
[data-testid="stSidebar"] label {
    color: white !important;
}
[data-testid="stSidebar"] input {
    color: black !important;
    background-color: #ecfdf5 !important;
}
.glass {
    background: rgba(255,255,255,0.85);
    padding: 30px;
    border-radius: 25px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    margin-top: 15px;
}
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 📥 SIDEBAR INPUTS
# =========================
st.sidebar.header("🌾 Input Soil Data")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0)
ph = st.sidebar.number_input("pH", min_value=0.0)
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
# 📑 TABS (FIXED ORDER)
# =========================
tab1, tab2, tab3 = st.tabs(["📘 About", "🌾 Prediction", "📊 Insights"])

# =========================
# 📘 TAB 1 → ABOUT
# =========================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📘 About This System</h3>

    <p>This system helps farmers and researchers choose the most suitable crop
    based on environmental and soil conditions, improving yield and decision-making.</p>

    <p>This AI system recommends crops using:</p>

    <ul>
    <li>🌱 Soil nutrients (N, P, K)</li>
    <li>🌡 Temperature</li>
    <li>💧 Humidity</li>
    <li>🧪 Soil pH</li>
    <li>🌧 Rainfall</li>
    </ul>

    <h4>🛠 How to Use</h4>
    <ul>
    <li>Enter values in the sidebar</li>
    <li>Click <b>Recommend Crop</b></li>
    <li>Check results in Prediction tab</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🌾 TAB 2 → PREDICTION
# =========================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    recommend = st.button("🌱 Recommend Crop", use_container_width=True)

    if recommend:
        if any(v <= 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
            st.warning("⚠️ Please enter all values greater than 0.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)

            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities

            crop = prediction[0]

            st.markdown(f"<div class='card'><h2>{crop}</h2></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📊 TAB 3 → INSIGHTS
# =========================
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if "prediction" in st.session_state:

        probabilities = st.session_state["probabilities"]

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })

        st.bar_chart(prob_df.set_index("Crop"))

        # 💡 SIMPLE EXPLANATION (NO SHAP)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 💡 Why this crop?")

        crop_explanations = {
            "rice": "Rice thrives in high rainfall and high humidity conditions.",
            "maize": "Maize grows well with moderate rainfall and balanced soil nutrients.",
            "coffee": "Coffee requires specific temperature ranges and consistent rainfall.",
        }

        crop_name = st.session_state["prediction"][0].lower()

        st.write(crop_explanations.get(
            crop_name,
            "This crop suits your soil and weather conditions."
        ))

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run a prediction first.")

    st.markdown('</div>', unsafe_allow_html=True)