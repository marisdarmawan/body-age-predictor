# 🤖 Prediktor Usia Tubuh (Body Age Predictor)

Proyek ini menggunakan model *machine learning* untuk memprediksi "usia tubuh" seseorang berdasarkan berbagai faktor kesehatan dan gaya hidup. Aplikasi ini dibangun menggunakan notebook Jupyter dan di-deploy sebagai aplikasi web interaktif dengan Streamlit.

---

## ✨ Fitur

* **Prediksi Usia**: Memberikan estimasi usia tubuh berdasarkan data input.
* **Analisis Data Komprehensif**: Melakukan pembersihan data, rekayasa fitur, dan visualisasi untuk memahami hubungan antar variabel.
* **Model Regresi**: Menggunakan `RandomForestRegressor` untuk tugas prediksi.
* **Antarmuka Web**: Disediakan file `app.py` untuk deployment menggunakan Streamlit, memungkinkan interaksi pengguna yang mudah.
* **Optimalisasi Model**: Termasuk proses tuning hyperparameter menggunakan `GridSearchCV` untuk meningkatkan akurasi model.

---

## ⚙️ Alur Kerja Proyek

1.  **Analisis & Pra-pemrosesan Data**:
    * Dataset dimuat dan dieksplorasi untuk menemukan *insight* awal.
    * **Rekayasa Fitur**: Kolom "Blood Pressure (s/d)" dipecah menjadi dua kolom terpisah: 'Systolic\_BP' dan 'Diastolic\_BP'.
    * **Pembersihan Data**: Nilai yang hilang (*missing values*) pada fitur numerik diisi dengan median, sedangkan pada fitur kategorikal diisi dengan modus.
    * **Encoding**: Fitur kategorikal seperti 'Gender' dan 'Smoking Status' diubah menjadi format numerik menggunakan `LabelEncoder`.
    * **Penskalaan Fitur**: Seluruh fitur numerik dinormalisasi menggunakan `MinMaxScaler` untuk memastikan skala data yang seragam.

2.  **Pembangunan Model**:
    * Model `RandomForestRegressor` dipilih untuk tugas prediksi ini.
    * Dilakukan *hyperparameter tuning* dengan `GridSearchCV` untuk menemukan kombinasi parameter terbaik, seperti `n_estimators`, `max_depth`, dan `min_samples_split`.
    * Model dievaluasi menggunakan metrik *Mean Squared Error* (MSE) pada data validasi.

3.  **Deployment**:
    * Model terbaik, scaler, dan label encoder disimpan sebagai file `pkl`.
    * File `app.py` disediakan untuk membuat aplikasi web interaktif di mana pengguna dapat memasukkan data mereka dan mendapatkan prediksi usia tubuh secara *real-time*.

---

## 🚀 Cara Menjalankan Aplikasi Streamlit

Untuk menjalankan aplikasi prediksi ini di komputer Anda, ikuti langkah-langkah berikut:

1.  **Pastikan Semua File Ada**:
    Simpan file-file berikut dalam satu direktori:
    * `app.py` (file aplikasi Streamlit)
    * `best_random_forest_model.pkl` (model yang telah dilatih)
    * `scaler.pkl` (scaler yang telah di-fit)
    * `label_encoders_dict.pkl` (encoder untuk variabel kategori)

2.  **Buat File `requirements.txt`**:
    Buat file bernama `requirements.txt` dan isi dengan library yang dibutuhkan:
    ```txt
    streamlit
    pandas
    numpy
    scikit-learn
    ```

3.  **Instal Ketergantungan**:
    Buka terminal atau command prompt, navigasi ke direktori proyek, dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**:
    Di terminal yang sama, jalankan perintah berikut:
    ```bash
    streamlit run app.py
    ```
    Aplikasi web akan otomatis terbuka di browser Anda.

---

# 🤖 Body Age Predictor

This project uses a *machine learning* model to predict a person's "body age" based on various health and lifestyle factors. The application is built using a Jupyter notebook and deployed as an interactive web app with Streamlit.

---

## ✨ Features

* **Age Prediction**: Provides an estimate of body age based on input data.
* **Comprehensive Data Analysis**: Performs data cleaning, feature engineering, and visualization to understand the relationships between variables.
* **Regression Model**: Uses `RandomForestRegressor` for the prediction task.
* **Web Interface**: An `app.py` file is provided for deployment using Streamlit, allowing for easy user interaction.
* **Model Optimization**: Includes a hyperparameter tuning process using `GridSearchCV` to improve model accuracy.

---

## ⚙️ Project Workflow

1.  **Data Analysis & Preprocessing**:
    * The dataset is loaded and explored to find initial insights.
    * **Feature Engineering**: The "Blood Pressure (s/d)" column is split into two separate columns: 'Systolic_BP' and 'Diastolic_BP'.
    * **Data Cleaning**: Missing values in numerical features are filled with the median, while categorical features are filled with the mode.
    * **Encoding**: Categorical features like 'Gender' and 'Smoking Status' are converted into a numerical format using `LabelEncoder`.
    * **Feature Scaling**: All numerical features are normalized using `MinMaxScaler` to ensure a uniform data scale.

2.  **Model Building**:
    * The `RandomForestRegressor` model was chosen for this prediction task.
    * *Hyperparameter tuning* was performed with `GridSearchCV` to find the best combination of parameters, such as `n_estimators`, `max_depth`, and `min_samples_split`.
    * The model is evaluated using the *Mean Squared Error* (MSE) metric on validation data.

3.  **Deployment**:
    * The best model, scaler, and label encoders are saved as `pkl` files.
    * An `app.py` file is provided to create an interactive web application where users can input their data and get a real-time body age prediction.

---

## 🚀 How to Run the Streamlit Application

To run this prediction application on your local machine, follow these steps:

1.  **Ensure All Files Are Present**:
    Save the following files in the same directory:
    * `app.py` (the Streamlit application file)
    * `best_random_forest_model.pkl` (the trained model)
    * `scaler.pkl` (the fitted scaler)
    * `label_encoders_dict.pkl` (the encoders for categorical variables)

2.  **Create a `requirements.txt` File**:
    Create a file named `requirements.txt` and add the required libraries to it:
    ```txt
    streamlit
    pandas
    numpy
    scikit-learn
    ```

3.  **Install Dependencies**:
    Open a terminal or command prompt, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    In the same terminal, run the following command:
    ```bash
    streamlit run app.py
    ```
    The web application will automatically open in your default browser.

---
