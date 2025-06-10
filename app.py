import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and encoders
with open('best_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders_dict.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Bode Age Predictor", layout="centered")

st.title("🧠 Bode Age Predictor")
st.write("Masukkan data untuk memprediksi usia berdasarkan indikator kesehatan.")

# Daftar fitur kategorikal
categorical_features = list(label_encoders.keys())

# Buat form input
with st.form("user_input_form"):
    user_input = {}

    # Input fitur kategorikal
    for col in categorical_features:
        options = label_encoders[col].classes_
        user_input[col] = st.selectbox(f"{col}", options)

    # Input fitur numerik
    numeric_features = [
        'Height (cm)', 'Weight (kg)', 'Cholesterol Level (mg/dL)',
        'BMI', 'Blood Glucose Level (mg/dL)', 'Bone Density (g/cm²)',
        'Vision Sharpness', 'Hearing Ability (dB)',
        'Systolic_BP', 'Diastolic_BP'
    ]
    for col in numeric_features:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    submitted = st.form_submit_button("Prediksi Usia")

if submitted:
    try:
        # Convert input ke DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode fitur kategorikal
        for col in categorical_features:
            encoder = label_encoders[col]
            input_df[col] = encoder.transform([input_df[col][0]])

        # Susun ulang urutan kolom jika model mengharuskan
        full_features = categorical_features + numeric_features
        input_df = input_df[full_features]

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Prediksi
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎉 Prediksi usia kamu adalah sekitar **{prediction:.1f} tahun**")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
