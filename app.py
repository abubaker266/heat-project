import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
import h5py
import json

# إعدادات الصفحة
st.set_page_config(page_title="Abubakar's Heat Predictor", layout="centered")

# دالة تحميل النموذج والموازين مع إصلاح الأخطاء تلقائياً
@st.cache_resource
def load_assets():
    model_path = "heat_model.h5"
    # هذا الجزء يحل مشكلة quantization_config التي ظهرت لك سابقاً
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
    
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
    return model, scaler_X, scaler_y

# واجهة المستخدم
try:
    model, scaler_X, scaler_y = load_assets()

    st.title("🔥 Heat Transfer over flat plat Predictor")
    st.markdown("<h3 style='text-align: center;'>Developed by: Abubakar</h3>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        V = st.number_input("Velocity (V)", value=1.0)
        Ts = st.number_input("Surface Temp (Ts)", value=350.0)
        L = st.number_input("Length (L)", value=0.5)
    with col2:
        T_inf = st.number_input("Ambient Temp (T_inf)", value=298.0)
        W = st.number_input("Width (W)", value=0.5)

    if st.button("RUN PREDICTION", type="primary"):
        # التوقع
        inputs = np.array([[V, Ts, T_inf, L, W]])
        inputs_scaled = scaler_X.transform(inputs)
        pred_scaled = model.predict(inputs_scaled)
        res = scaler_y.inverse_transform(pred_scaled)

        # عرض q و h
        st.markdown("### 📊 Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Heat Flux (q)", f"{res[0][0]:.4f} W/m²")
        res_col2.metric("Coeff (h)", f"{res[0][1]:.4f} W/m²·K")
        st.balloons()

except Exception as e:
    st.error(f"Error: {e}")
