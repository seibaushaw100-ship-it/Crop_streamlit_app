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

/* Background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #f0fdf4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}

/* Sidebar text fix */
[data-testid="stSidebar"] label {
    color: white !important;
}

/* Fix input visibility */
[data-testid="stSidebar"] input {
    color: black !important;
    background-color: #ecfdf5 !important;
}

/* Glass container */
.glass {
    background: rgba(255,255,255,0.85);
    padding: 30px;
    border-radius: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-top: 15px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #15803d);
    color: white;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    box-shadow: 0 6px 15px rgba(34,197,94,0.4);
}

/* Inputs */
.stNumberInput input {
    background-color: #ecfdf5;
    border-radius: 10px;
    border: 1px solid #bbf7d0;
}

/* Title */
h1 {
    text-align: center;
    color: #022c22;
}

/* 📱 Mobile Optimization */
@media (max-width: 768px) {

    .glass {
        padding: 15px !important;
        border-radius: 15px !important;
    }

    .card {
        padding: 15px !important;
    }

    h1 {
        font-size: 22px !important;
    }

    .stButton>button {
        font-size: 16px !important;
        padding: 10px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# ⚡ LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
    return model, scaler

model, scaler = load_model()

# =========================
# 🧠 LOAD SHAP EXPLAINER (FIXED)
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
# 📊 INPUTS (SIDEBAR)
# =========================
st.sidebar.header("🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, value=50.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, value=50.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0, value=50.0)
temperature = st.sidebar.number_input("Temperature (°C)", value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
ph = st.sidebar.number_input("Soil pH", value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=100.0)

# =========================
# 🎯 CROP ICONS
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
# 🌾 TAB 1: PREDICTION
# =========================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 🌿 Run Prediction")
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

            # Save for other tabs
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
# 📊 TAB 2: INSIGHTS
# =========================
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if "prediction" in st.session_state:

        probabilities = st.session_state["probabilities"]

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })

        # Top 5 chart
        top_df = prob_df.sort_values(by="Probability", ascending=False).head(5)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📊 Top Predictions")
        st.bar_chart(top_df.set_index("Crop"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 3 text
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🌾 Top 3 Crops")

        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"{model.classes_[i]} → {probs[i]*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # SHAP Explainability
        if "shap_values" in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### 🧠 Why this prediction?")

            shap_values = st.session_state["shap_values"]

            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(bbox_inches='tight')

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run a prediction first to see insights.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📘 TAB 3: ABOUT
# =========================
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📘 About This System</h3>

    <p>
    This AI-powered system recommends the best crops based on:
    </p>

    <ul>
    <li>🌱 Soil nutrients (N, P, K)</li>
    <li>🌡 Temperature</li>
    <li>💧 Humidity</li>
    <li>🧪 Soil pH</li>
    <li>🌧 Rainfall</li>
    </ul>

    <p>
    Built using <b>Machine Learning</b>, <b>Streamlit</b>, and <b>Explainable AI (SHAP)</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)