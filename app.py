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

# ✅ FULL UI STYLING
st.markdown("""
<style>

/* 🌈 Background */
.stApp {
    background: linear-gradient(135deg, #dbeafe, #f0fdf4);
}

/* Main spacing */
.block-container {
    padding-top: 2rem;
}

/* 🧊 Main centered container */
.main-card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* 🌿 Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #065f46, #022c22);
}
[data-testid="stSidebar"] * {
    color: white;
}

/* 🔘 Button */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.03);
}

/* 📦 Inner cards */
.card {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
}

/* Titles */
h1 {
    text-align: center;
    color: #064e3b;
}
h3 {
    color: #065f46;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ✅ Load model
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))

# ✅ Title
st.markdown("<h1>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter soil and weather conditions to get the best crop recommendation</p>", unsafe_allow_html=True)

# ✅ START MAIN CONTAINER
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ✅ Sidebar Inputs
st.sidebar.header("🌍 Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0.0)
P = st.sidebar.number_input("Phosphorus (P)", 0.0)
K = st.sidebar.number_input("Potassium (K)", 0.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0)
ph = st.sidebar.number_input("Soil pH", 0.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0)

st.markdown("<br>", unsafe_allow_html=True)

# ✅ CENTERED BUTTON
col1, col2, col3 = st.columns([1,2,1])
with col2:
    recommend = st.button("🌱 Recommend Crop", use_container_width=True)

# ✅ Prediction Logic
if recommend:

    if N == 0 or P == 0 or K == 0:
        st.warning("⚠️ Please enter valid soil nutrient values.")
    else:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        confidence = np.max(probabilities) * 100

        # 🌾 Result
        st.markdown(f"""
        <div class="card" style="text-align:center; font-size:26px;">
            🌾 <b>{prediction[0]}</b>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"📊 Confidence Level: {confidence:.2f}%")

        # 🌾 Top 3
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🌾 Top 3 Recommended Crops:")
        probs = probabilities[0]
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"**{model.classes_[i]}** → {probs[i]*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # 💡 Explanation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 💡 Why this crop?")
        if prediction[0] == "rice":
            st.write("Rice thrives in high rainfall and high humidity conditions.")
        elif prediction[0] == "maize":
            st.write("Maize grows well in moderate rainfall and balanced soil nutrients.")
        elif prediction[0] == "coffee":
            st.write("Coffee requires moderate rainfall and specific temperature ranges.")
        else:
            st.write("This crop matches the given conditions.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 📊 Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📊 Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": probabilities[0]
        })

        st.bar_chart(prob_df.set_index("Crop"))
        st.markdown('</div>', unsafe_allow_html=True)

# 📘 About
st.markdown("""
<div class="card">
<h3>📘 About This Project</h3>
<p>This AI-powered system recommends the best crops based on soil nutrients and weather conditions using machine learning models.</p>
</div>
""", unsafe_allow_html=True)

# ✅ END MAIN CONTAINER
st.markdown('</div>', unsafe_allow_html=True)