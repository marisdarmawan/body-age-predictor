import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- FUNGSI UNTUK MEMUAT ARTIFAK ---
# Menggunakan cache untuk memuat model, scaler, dan encoder hanya sekali.
@st.cache_resource
def load_artifacts():
    """
    Fungsi ini memuat model, scaler, dan label encoder yang telah disimpan dari file pickle.
    """
    try:
        with open('best_random_forest_model.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        with open('scaler.pkl', 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        with open('label_encoders_dict.pkl', 'rb') as f_le:
            label_encoders = pickle.load(f_le)
    except FileNotFoundError:
        st.error("File model atau artefak tidak ditemukan. Pastikan file 'best_random_forest_model.pkl', 'scaler.pkl', dan 'label_encoders_dict.pkl' berada di direktori yang sama dengan app.py.")
        st.stop()
    return model, scaler, label_encoders

# --- MEMUAT MODEL, SCALER, DAN ENCODER ---
model, scaler, label_encoders = load_artifacts()

# --- JUDUL DAN DESKRIPSI APLIKASI ---
st.title("Prediktor Usia Tubuh (Body Age Predictor)")
st.write("""
Aplikasi ini memprediksi "usia tubuh" Anda berdasarkan berbagai faktor kesehatan dan gaya hidup.
Masukkan data Anda di sidebar untuk melihat hasilnya.
""")

# --- SIDEBAR UNTUK INPUT PENGGUNA ---
st.sidebar.header("Masukkan Data Anda")

def user_input_features():
    """
    Fungsi ini membuat semua widget input di sidebar dan mengembalikan
    input pengguna sebagai sebuah pandas DataFrame.
    """
    input_data = {}

    # Mengambil urutan fitur yang benar langsung dari model yang telah dilatih
    # Ini sangat penting untuk menghindari kesalahan urutan kolom
    all_features_order = model.feature_names_in_

    # Loop melalui semua fitur sesuai urutan yang benar
    for feature in all_features_order:
        # Input untuk fitur kategorikal (menggunakan selectbox dengan istilah asli)
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            input_data[feature] = st.sidebar.selectbox(f"Pilih {feature}", options)
        # Input untuk fitur numerik
        else:
            # Memberikan nilai default yang lebih masuk akal untuk setiap fitur
            default_value = 0.0
            if "BP" in feature:
                default_value = 120.0
            elif "Height" in feature:
                default_value = 170.0
            elif "Weight" in feature:
                default_value = 70.0
            elif "BMI" in feature:
                default_value = 22.0
            elif "Cholesterol" in feature or "Glucose" in feature:
                default_value = 150.0
            elif "Density" in feature or "Sharpness" in feature:
                default_value = 0.8
            input_data[feature] = st.sidebar.number_input(f"Masukkan {feature}", value=default_value)

    return pd.DataFrame([input_data])

# --- MENGAMBIL INPUT DARI PENGGUNA ---
input_df = user_input_features()

# --- MENAMPILKAN DATA INPUT (OPSIONAL) ---
st.subheader("Data Input Anda")
st.write(input_df)

# --- TOMBOL PREDIKSI ---
if st.sidebar.button("Prediksi Usia Tubuh"):
    if model and scaler and label_encoders:
        # 1. BUAT SALINAN DATA INPUT UNTUK DIPROSES
        processed_df = input_df.copy()

        # 2. ENCODE FITUR KATEGORIKAL
        # Mengubah input string (misal: "Pria") menjadi angka (misal: 1)
        for feature, le in label_encoders.items():
            if feature in processed_df.columns:
                processed_df[feature] = le.transform(processed_df[feature])

        # 3. PASTIKAN URUTAN KOLOM SESUAI DENGAN DATA PELATIHAN
        # Ini adalah langkah kunci untuk memperbaiki error ketidakcocokan fitur
        try:
            ordered_df = processed_df[model.feature_names_in_]
        except KeyError as e:
            st.error(f"Terjadi kesalahan urutan kolom: {e}. Pastikan semua fitur yang dibutuhkan ada.")
            st.stop()

        # 4. SKALAKAN SEMUA FITUR MENGGUNAKAN SCALER YANG TELAH DISIMPAN
        # Scaler diterapkan pada seluruh DataFrame, persis seperti di notebook
        scaled_input = scaler.transform(ordered_df)

        # 5. BUAT PREDIKSI
        prediction = model.predict(scaled_input)

        # 6. TAMPILKAN HASIL PREDIKSI
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi usia tubuh Anda adalah: **{int(prediction[0])} tahun**")
    else:
        st.warning("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")
