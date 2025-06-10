import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- FUNGSI UNTUK MEMUAT ARTIFAK ---
# Menggunakan cache untuk meningkatkan performa.
@st.cache_resource
def load_artifacts():
    """
    Memuat model, scaler, dan label encoder yang telah disimpan dari file pickle.
    """
    try:
        with open('best_random_forest_model.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        with open('scaler.pkl', 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        with open('label_encoders_dict.pkl', 'rb') as f_le:
            label_encoders = pickle.load(f_le)
    except FileNotFoundError as e:
        st.error(f"File artefak tidak ditemukan: {e}. Pastikan file .pkl berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat artefak: {e}")
        st.stop()
    return model, scaler, label_encoders

# --- Memuat Artefak ---
model, scaler, label_encoders = load_artifacts()

# --- Judul dan Deskripsi Aplikasi ---
st.title("Prediktor Usia Tubuh (Body Age Predictor)")
st.write("Masukkan fitur-fitur Anda di sidebar untuk memprediksi usia.")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Input Fitur")

def user_input_features():
    """
    Membuat semua widget input dan mengembalikan input dalam bentuk DataFrame.
    """
    input_data = {}
    
    # Kumpulkan fitur kategori dan numerik sesuai notebook
    categorical_features = ['Gender', 'Smoking Status', 'Sleep Patterns', 'Diet', 'Physical Activity Level', 'Income Level', 'Mental Health Status']
    all_features_order = model.feature_names_in_ # Urutan fitur yang benar dari model

    # Buat input untuk setiap fitur
    for feature in all_features_order:
        if feature in categorical_features:
            # Dapatkan opsi dari label encoder yang sudah di-fit
            options = list(label_encoders[feature].classes_)
            input_data[feature] = st.sidebar.selectbox(f"Pilih {feature}", options)
        else: # Fitur numerik
            # Berikan nilai default yang masuk akal
            default_value = 0.0
            if "BP" in feature or "Cholesterol" in feature or "Glucose" in feature:
                default_value = 120.0
            elif "Height" in feature:
                default_value = 170.0
            elif "Weight" in feature:
                default_value = 70.0
            elif "BMI" in feature:
                default_value = 22.0
            input_data[feature] = st.sidebar.number_input(f"Masukkan {feature}", value=default_value)
            
    return pd.DataFrame([input_data])

# Kumpulkan input dari pengguna
input_df = user_input_features()

# Tampilkan input pengguna (opsional)
st.subheader("Input yang Anda Masukkan:")
st.write(input_df)

# --- Tombol Prediksi ---
if st.sidebar.button("Prediksi Usia"):
    
    # 1. Buat salinan DataFrame input untuk diproses
    processed_df = input_df.copy()

    # 2. Encode fitur kategori
    # Loop melalui semua fitur kategori yang didefinisikan di notebook
    categorical_to_encode = ['Gender', 'Smoking Status', 'Sleep Patterns', 'Diet', 'Physical Activity Level', 'Income Level', 'Mental Health Status']
    for feature in categorical_to_encode:
        # Cek jika fitur ada di DataFrame sebelum melakukan transform
        if feature in processed_df.columns:
            try:
                le = label_encoders[feature]
                processed_df[feature] = le.transform(processed_df[feature])
            except ValueError:
                st.error(f"Nilai untuk fitur '{feature}' tidak dikenali. Harap pilih dari opsi yang tersedia.")
                st.stop()

    # 3. PASTIKAN URUTAN KOLOM SUDAH BENAR
    # Ini adalah langkah paling penting untuk memperbaiki error
    try:
        ordered_df = processed_df[model.feature_names_in_]
    except KeyError as e:
        st.error(f"Terjadi kesalahan urutan kolom: {e}. Pastikan semua fitur yang dibutuhkan ada.")
        st.stop()

    # 4. Skalakan semua fitur menggunakan scaler yang sama
    scaled_input = scaler.transform(ordered_df)

    # 5. Buat prediksi
    prediction = model.predict(scaled_input)

    # 6. Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.success(f"Prediksi usia Anda adalah: {prediction[0]:.0f} tahun")

st.markdown("---")
st.write("Aplikasi Streamlit sederhana untuk prediksi usia.")
