```python
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# ✅ Page config
st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌱",
    layout="wide"
)

# ✅ UI Styling (Dashboard Level)
st.markdown("""
<style>

/* 🌈 Background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #f0fdf4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}
[data-testid="stSidebar"] * {
    color: white;
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

</style>
""", unsafe_allow_html=True)

# ✅ Load model
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))

# ✅ Title
st.markdown("<h1>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# ✅ Sidebar Inputs
st.sidebar.header("🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0.0)
P = st.sidebar.number_input("Phosphorus (P)", 0.0)
K = st.sidebar.number_input("Potassium (K)", 0.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0)
ph = st.sidebar.number_input("Soil pH", 0.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0)

# ✅ Tabs
tab1, tab2, tab3 = st.tabs(["🌾 Prediction", "📊 Insights", "📘 About"])

# =========================
# 🌾 TAB 1: PREDICTION
# =========================
with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        recommend = st.button("🌱 Recommend Crop", use_container_width=True)

    if recommend:
        if N == 0 or P == 0 or K == 0:
            st.warning("⚠️ Please enter valid soil nutrient values.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            confidence = np.max(probabilities) * 100

            # Save to session for other tabs
            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities

            st.markdown(f"""
            <div class="card" style="text-align:center; font-size:26px;">
                🌾 <b>{prediction[0]}</b>
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

        # Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📊 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Crop"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 3
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🌾 Top 3 Crops")

        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"{model.classes_[i]} → {probs[i]*100:.2f}%")

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
    <h3>📘 About This Project</h3>
    <p>This AI-powered system recommends the best crops based on soil nutrients and weather conditions using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
```
