import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
import h5py
import json

# 1. Page Config: Sets the browser tab title and layout
st.set_page_config(page_title="Abubakar's Heat Predictor", layout="centered")

# 2. Custom CSS: Changes colors (Red button, Blue titles)
st.markdown("""
    <style>
    .stButton>button { background-color: #ff4b2b; color: white; border-radius: 8px; width: 100%; }
    h1 { color: #1f77b4; text-align: center; }
    .author { text-align: center; font-weight: bold; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets: Reads the AI model and the Scalers
@st.cache_resource
def load_assets():
    model_path = "heat_model.h5"
    # This block fixes the 'quantization_config' error for the server
    try:
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                def clean(obj):
                    if isinstance(obj, dict):
                        obj.pop('quantization_config', None)
                        for k,v in obj.items(): clean(v)
                    elif isinstance(obj, list):
                        for i in obj: clean(i)
                clean(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
    except: pass
    
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
    return model, scaler_X, scaler_y

# 4. Main App Logic
try:
    model, scaler_X, scaler_y = load_assets()

    st.title("🔥 Heat Transfer Predictor")
    st.markdown("<p class='author'>Developed by: Abubaker mukhtar</p>", unsafe_allow_html=True)
    st.divider()

    # Input Fields for the 5 parameters
    col1, col2 = st.columns(2)
    with col1:
        V = st.number_input("Velocity (V)", value=1.0)
        Ts = st.number_input("Surface Temp (Ts)", value=350.0)
        L = st.number_input("Length (L)", value=0.5)
    with col2:
        T_inf = st.number_input("Ambient Temp (T_inf)", value=298.0)
        W = st.number_input("Width (W)", value=0.5)

    if st.button("RUN PREDICTION"):
        # Convert inputs to Array -> Scale -> Predict -> Inverse Scale
        inputs = np.array([[V, Ts, T_inf, L, W]])
        inputs_scaled = scaler_X.transform(inputs)
        pred_scaled = model.predict(inputs_scaled)
        res = scaler_y.inverse_transform(pred_scaled)

        # Show Results (q and h)
        st.markdown("### 📊 Prediction Results")
        c1, c2 = st.columns(2)
        c1.metric("Heat Flux (q)", f"{res[0][0]:.4f} W/m²")
        c2.metric("Convection Coeff (h)", f"{res[0][1]:.4f} W/m²·K")
        st.balloons()

except Exception as e:
    st.error(f"Error: {e}")