import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import shap

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
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-top: 15px;
}

.stButton>button {
    background: linear-gradient(90deg, #22c55e, #15803d);
    color: white;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
}

.stNumberInput input {
    background-color: #ecfdf5;
    border-radius: 10px;
}

h1 {
    text-align: center;
    color: #022c22;
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
# 🧠 LOAD SHAP
# =========================
@st.cache_resource
def load_explainer():
    return shap.Explainer(model)

explainer = load_explainer()

# =========================
# 🌱 TITLE
# =========================
st.markdown("<h1>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# =========================
# 🎯 ICONS
# =========================
crop_icons = {
    "rice": "🌾",
    "maize": "🌽",
    "apple": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "orange": "🍊",
    "cotton": "🧵",
    "coffee": "☕"
}

# =========================
# 📑 TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🌾 Prediction", "📊 Insights", "📘 About"])

# =========================
# 🌾 TAB 1
# =========================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🌿 Run Prediction")
    recommend = st.button("🌱 Recommend Crop", use_container_width=True)

    if recommend:
        if any(v <= 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
            st.warning("⚠️ Please enter all values greater than 0.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)

            with st.spinner("Analyzing soil data... 🌱"):
                prediction = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)
                shap_values = explainer(input_scaled)

            confidence = np.max(probabilities) * 100

            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities
            st.session_state["shap_values"] = shap_values

            crop = prediction[0]
            icon = crop_icons.get(crop.lower(), "🌱")

            st.markdown(f"""
            <div class="card" style="text-align:center; font-size:28px;">
                {icon} <b>{crop}</b>
            </div>
            """, unsafe_allow_html=True)

            st.success(f"Confidence: {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📊 TAB 2
# =========================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if "prediction" in st.session_state:

        probabilities = st.session_state["probabilities"]

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })

        # Top 5
        top_df = prob_df.sort_values(by="Probability", ascending=False).head(5)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📊 Top Predictions")
        st.bar_chart(top_df.set_index("Crop"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 3
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🌾 Top 3 Crops")

        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"{model.classes_[i]} → {probs[i]*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # SHAP
        if "shap_values" in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### 🧠 Why this prediction?")

            shap_values = st.session_state["shap_values"]

            values = shap_values.values

            if len(values.shape) == 3:
                values = values[0, :, 0]
            else:
                values = values[0]

            feature_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": values
            }).sort_values(by="Impact", key=abs, ascending=False)

            st.write("Positive = increases prediction, Negative = decreases")
            st.bar_chart(shap_df.set_index("Feature"))

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run a prediction first to see insights.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📘 TAB 3
# =========================
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📘 About This System</h3>

    <p>This AI system recommends crops using:</p>

    <ul>
    <li>🌱 Soil nutrients (N, P, K)</li>
    <li>🌡 Temperature</li>
    <li>💧 Humidity</li>
    <li>🧪 Soil pH</li>
    <li>🌧 Rainfall</li>
    </ul>

    <p>
    Built with <b>Machine Learning</b>, <b>Streamlit</b>, and <b>SHAP</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)