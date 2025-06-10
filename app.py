import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- FUNGSI UNTUK MEMUAT MODEL DAN ARTIFAK LAINNYA ---
# Menggunakan cache untuk meningkatkan performa dengan tidak memuat ulang file pada setiap interaksi.
@st.cache_resource
def load_artifacts():
    """
    Fungsi ini memuat model, scaler, dan label encoder yang telah disimpan.
    """
    try:
        with open('best_random_forest_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        with open('label_encoders_dict.pkl', 'rb') as le_file:
            label_encoders = pickle.load(le_file)
    except FileNotFoundError:
        st.error("File model atau artefak tidak ditemukan. Pastikan file 'best_random_forest_model.pkl', 'scaler.pkl', dan 'label_encoders_dict.pkl' berada di direktori yang sama dengan app.py.")
        return None, None, None
    return model, scaler, label_encoders

# --- MEMUAT MODEL ---
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
    Fungsi ini membuat semua widget input di sidebar dan mengembalikan input pengguna
    sebagai sebuah pandas DataFrame.
    """
    # Mendapatkan kelas dari label encoder untuk dropdown
    gender_options = label_encoders['Gender'].classes_
    smoking_options = label_encoders['Smoking Status'].classes_
    sleep_options = label_encoders['Sleep Patterns'].classes_
    diet_options = label_encoders['Diet'].classes_
    activity_options = label_encoders['Physical Activity Level'].classes_
    income_options = label_encoders['Income Level'].classes_
    mental_health_options = label_encoders['Mental Health Status'].classes_

    # Membuat widget input
    gender = st.sidebar.selectbox("Jenis Kelamin (Gender)", gender_options)
    height_cm = st.sidebar.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight_kg = st.sidebar.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
    systolic_bp = st.sidebar.number_input("Tekanan Darah Sistolik (BP)", min_value=80, max_value=200, value=120)
    diastolic_bp = st.sidebar.number_input("Tekanan Darah Diastolik (BP)", min_value=50, max_value=120, value=80)
    cholesterol = st.sidebar.number_input("Tingkat Kolesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0)
    bmi = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", min_value=15.0, max_value=50.0, value=22.0)
    glucose = st.sidebar.number_input("Tingkat Glukosa Darah (mg/dL)", min_value=50.0, max_value=250.0, value=90.0)
    bone_density = st.sidebar.number_input("Kepadatan Tulang (g/cm²)", min_value=0.1, max_value=2.0, value=0.8)
    vision = st.sidebar.number_input("Ketajaman Penglihatan (0-1)", min_value=0.0, max_value=1.0, value=0.8)
    hearing = st.sidebar.number_input("Kemampuan Pendengaran (dB)", min_value=10.0, max_value=100.0, value=30.0)
    activity = st.sidebar.selectbox("Tingkat Aktivitas Fisik", activity_options)
    smoking = st.sidebar.selectbox("Status Merokok", smoking_options)
    diet = st.sidebar.selectbox("Jenis Diet", diet_options)
    cognitive = st.sidebar.number_input("Fungsi Kognitif (0-100)", min_value=0.0, max_value=100.0, value=80.0)
    mental_health = st.sidebar.selectbox("Status Kesehatan Mental", mental_health_options)
    sleep = st.sidebar.selectbox("Pola Tidur", sleep_options)
    stress = st.sidebar.number_input("Tingkat Stres (0-10)", min_value=0.0, max_value=10.0, value=3.0)
    pollution = st.sidebar.number_input("Paparan Polusi (0-10)", min_value=0.0, max_value=10.0, value=4.0)
    sun_exposure = st.sidebar.number_input("Paparan Matahari (jam/minggu)", min_value=0.0, max_value=40.0, value=5.0)
    income = st.sidebar.selectbox("Tingkat Pendapatan", income_options)
    
    # Mengumpulkan data ke dalam dictionary
    data = {
        'Gender': gender,
        'Height (cm)': height_cm,
        'Weight (kg)': weight_kg,
        'Cholesterol Level (mg/dL)': cholesterol,
        'BMI': bmi,
        'Blood Glucose Level (mg/dL)': glucose,
        'Bone Density (g/cm²)': bone_density,
        'Vision Sharpness': vision,
        'Hearing Ability (dB)': hearing,
        'Physical Activity Level': activity,
        'Smoking Status': smoking,
        'Diet': diet,
        'Cognitive Function': cognitive,
        'Mental Health Status': mental_health,
        'Sleep Patterns': sleep,
        'Stress Levels': stress,
        'Pollution Exposure': pollution,
        'Sun Exposure': sun_exposure,
        'Income Level': income,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp
    }
    
    # Mengonversi dictionary ke pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# --- MENGAMBIL INPUT ---
input_df = user_input_features()

# --- MENAMPILKAN INPUT PENGGUNA (OPSIONAL) ---
st.subheader("Data Input Anda")
st.write(input_df)

# --- TOMBOL PREDIKSI ---
if st.button("Prediksi Usia Tubuh"):
    if model is not None and scaler is not None and label_encoders is not None:
        # 1. PRA-PEMROSESAN INPUT
        # Buat salinan untuk pra-pemrosesan
        processed_df = input_df.copy()
        
        # Mengkodekan fitur kategori
        for feature, le in label_encoders.items():
            processed_df[feature] = le.transform(processed_df[feature])
            
        # 2. MEMASTIKAN URUTAN KOLOM SESUAI DENGAN MODEL
        # Mengambil urutan kolom dari model yang telah dilatih
        model_columns = model.feature_names_in_
        processed_df = processed_df[model_columns]
        
        # 3. MENSKALAKAN FITUR
        # Menggunakan scaler yang sama dari notebook
        scaled_features = scaler.transform(processed_df)
        
        # 4. MEMBUAT PREDIKSI
        prediction = model.predict(scaled_features)
        
        # 5. MENAMPILKAN HASIL
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi Usia Tubuh Anda adalah: **{int(prediction[0])} tahun**")
    else:
        st.warning("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")

