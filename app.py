import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- FUNGSI UNTUK MEMUAT MODEL DAN ARTIFAK LAINNYA ---
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
st.title("Prediktor Usia Tubuh (Body Age Predictor) 🩺")
st.write("""
Aplikasi ini memprediksi "usia tubuh" Anda berdasarkan berbagai faktor kesehatan dan gaya hidup. 
Masukkan data Anda di panel sebelah kiri untuk melihat hasilnya.
""")

# --- SIDEBAR UNTUK INPUT PENGGUNA ---
st.sidebar.header("📝 Masukkan Data Anda")

def user_input_features():
    """
    Fungsi ini membuat semua widget input di sidebar dan mengembalikan input pengguna
    sebagai sebuah pandas DataFrame dengan label yang lebih user-friendly.
    """
    # PENTING: Sesuaikan nilai di KANAN (e.g., 'Female', 'Male') agar SAMA PERSIS
    # dengan nilai yang digunakan saat melatih model Anda.
    # Nilai di KIRI adalah teks yang akan tampil di aplikasi.
    
    # --- Pemetaan Label untuk Tampilan yang Lebih Baik ---
    gender_map = {'Wanita': 'Female', 'Pria': 'Male'}
    smoking_map = {'Tidak Pernah': 'Non-smoker', 'Mantan Perokok': 'Former smoker', 'Masih Merokok': 'Current smoker'}
    activity_map = {'Jarang (Sedentary)': 'Sedentary', 'Cukup Aktif (Moderate)': 'Moderate', 'Sangat Aktif (Active)': 'Active'}
    diet_map = {'Seimbang': 'Balanced', 'Kurang Sehat': 'Unhealthy', 'Sehat': 'Healthy'}
    sleep_map = {'Baik (7-9 jam)': 'Good', 'Cukup (5-6 jam)': 'Fair', 'Kurang (<5 jam)': 'Poor'}
    income_map = {'Rendah': 'Low', 'Menengah': 'Middle', 'Tinggi': 'High'}
    mental_health_map = {'Baik': 'Good', 'Cukup': 'Fair', 'Buruk': 'Poor'}

    # --- Membuat Widget Input ---
    # Menggunakan st.radio untuk pilihan biner/sedikit agar lebih cepat
    gender_display = st.sidebar.radio("Jenis Kelamin", options=list(gender_map.keys()))
    
    # Input numerik
    height_cm = st.sidebar.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight_kg = st.sidebar.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
    systolic_bp = st.sidebar.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120, help="Angka atas pada pengukuran tekanan darah.")
    diastolic_bp = st.sidebar.number_input("Tekanan Darah Diastolik", min_value=50, max_value=120, value=80, help="Angka bawah pada pengukuran tekanan darah.")
    cholesterol = st.sidebar.number_input("Kolesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0)
    bmi = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", min_value=15.0, max_value=50.0, value=22.0)
    glucose = st.sidebar.number_input("Glukosa Darah (mg/dL)", min_value=50.0, max_value=250.0, value=90.0)
    bone_density = st.sidebar.number_input("Kepadatan Tulang (g/cm²)", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    vision = st.sidebar.number_input("Ketajaman Penglihatan (0-1)", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    hearing = st.sidebar.number_input("Kemampuan Pendengaran (dB)", min_value=10.0, max_value=100.0, value=30.0)
    cognitive = st.sidebar.number_input("Fungsi Kognitif (0-100)", min_value=0.0, max_value=100.0, value=80.0)
    stress = st.sidebar.slider("Tingkat Stres (0-10)", min_value=0.0, max_value=10.0, value=3.0)
    pollution = st.sidebar.slider("Paparan Polusi (0-10)", min_value=0.0, max_value=10.0, value=4.0)
    sun_exposure = st.sidebar.number_input("Paparan Matahari (jam/minggu)", min_value=0.0, max_value=40.0, value=5.0)

    # Input kategori menggunakan selectbox dengan label deskriptif
    smoking_display = st.sidebar.selectbox("Status Merokok", options=list(smoking_map.keys()))
    activity_display = st.sidebar.selectbox("Tingkat Aktivitas Fisik", options=list(activity_map.keys()))
    diet_display = st.sidebar.selectbox("Kualitas Diet/Pola Makan", options=list(diet_map.keys()))
    sleep_display = st.sidebar.selectbox("Kualitas Pola Tidur", options=list(sleep_map.keys()))
    mental_health_display = st.sidebar.selectbox("Status Kesehatan Mental", options=list(mental_health_map.keys()))
    income_display = st.sidebar.selectbox("Tingkat Pendapatan", options=list(income_map.keys()))
    
    # --- Mengumpulkan Data ke Dictionary ---
    # Menggunakan pemetaan untuk mengubah input display kembali ke nilai asli untuk model
    data = {
        'Gender': gender_map[gender_display],
        'Height (cm)': height_cm,
        'Weight (kg)': weight_kg,
        'Cholesterol Level (mg/dL)': cholesterol,
        'BMI': bmi,
        'Blood Glucose Level (mg/dL)': glucose,
        'Bone Density (g/cm²)': bone_density,
        'Vision Sharpness': vision,
        'Hearing Ability (dB)': hearing,
        'Physical Activity Level': activity_map[activity_display],
        'Smoking Status': smoking_map[smoking_display],
        'Diet': diet_map[diet_display],
        'Cognitive Function': cognitive,
        'Mental Health Status': mental_health_map[mental_health_display],
        'Sleep Patterns': sleep_map[sleep_display],
        'Stress Levels': stress,
        'Pollution Exposure': pollution,
        'Sun Exposure': sun_exposure,
        'Income Level': income_map[income_display],
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp
    }
    
    # Mengonversi dictionary ke pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# --- MENGAMBIL INPUT ---
input_df = user_input_features()

# --- MENAMPILKAN INPUT PENGGUNA (OPSIONAL) ---
# Menampilkan nilai asli yang akan diproses, bukan nilai display
st.subheader("Data Input Anda (Nilai Asli)")
st.write(input_df)

# --- TOMBOL PREDIKSI ---
if st.button("Prediksi Usia Tubuh", type="primary"):
    if model is not None and scaler is not None and label_encoders is not None:
        try:
            # 1. PRA-PEMROSESAN INPUT
            processed_df = input_df.copy()
            
            # Mengkodekan fitur kategori
            for feature in label_encoders:
                # Memastikan kolom ada di dataframe sebelum diproses
                if feature in processed_df.columns:
                    le = label_encoders[feature]
                    # Mengatasi kemungkinan nilai baru yang tidak ada saat training
                    # Dengan cara mengambil nilai dari map yang sudah pasti ada di classes_
                    processed_df[feature] = le.transform(processed_df[feature])
            
            # 2. MEMASTIKAN URUTAN KOLOM SESUAI DENGAN MODEL
            model_columns = model.feature_names_in_
            processed_df = processed_df[model_columns]
            
            # 3. MENSKALAKAN FITUR
            scaled_features = scaler.transform(processed_df)
            
            # 4. MEMBUAT PREDIKSI
            prediction = model.predict(scaled_features)
            
            # 5. MENAMPILKAN HASIL
            st.subheader("✨ Hasil Prediksi ✨")
            st.success(f"Prediksi Usia Tubuh Anda adalah: **{int(prediction[0])} tahun**")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.warning("Pastikan nilai pada pemetaan label (map) di dalam kode sesuai dengan data training.")
    else:
        st.warning("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")
