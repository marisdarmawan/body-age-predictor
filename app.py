import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model, scaler, and label encoders
try:
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders_dict.pkl', 'rb') as f:
        label_encoders_dict = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'best_random_forest_model.pkl', 'scaler.pkl', and 'label_encoders_dict.pkl' are in the same directory.")
    st.stop()

st.title("Human Age Prediction")
st.write("Enter the features to predict the age.")

# Create input fields for each feature
st.sidebar.header("Feature Input")

# Define the features and their types/options
# You need to list all the features your model was trained on in the correct order
# Refer to X.columns from your Colab notebook
features = {
    'Gender': ['Male', 'Female'],
    'Height (cm)': 'number',
    'Weight (kg)': 'number',
    'Cholesterol Level (mg/dL)': 'number',
    'BMI': 'number',
    'Blood Glucose Level (mg/dL)': 'number',
    'Bone Density (g/cm²)': 'number',
    'Vision Sharpness': 'number',
    'Hearing Ability (dB)': 'number',
    'Physical Activity Level': ['Low', 'Moderate', 'High'],
    'Smoking Status': ['Never', 'Former', 'Current'],
    'Diet': ['Balanced', 'High Protein', 'Low Carb', 'Vegetarian', 'Mediterranean'], # Add other diet types if present
    'Cognitive Function': 'number',
    'Mental Health Status': ['Good', 'Poor', 'Average'], # Add other mental health statuses if present
    'Sleep Patterns': ['Normal', 'Insomnia', 'Early Riser'], # Add other sleep patterns if present
    'Stress Levels': 'number',
    'Pollution Exposure': 'number',
    'Sun Exposure': 'number',
    'Income Level': ['Low', 'Medium', 'High'], # Add other income levels if present
    'Systolic_BP': 'number',
    'Diastolic_BP': 'number'
}

input_data = {}
for feature, options in features.items():
    if isinstance(options, list):
        input_data[feature] = st.sidebar.selectbox(f"Select {feature}", options)
    else:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0) # Set a default value

# Preprocess the input data
def preprocess_input(data, scaler, label_encoders):
    df = pd.DataFrame([data])

    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature in df.columns:
            # Handle potential unseen labels by using a try-except block or checking classes
            try:
                 df[feature] = encoder.transform(df[feature])
            except ValueError as e:
                 st.warning(f"Could not encode feature '{feature}': {e}. This might be an unseen category.")
                 # Handle unseen categories - you might need a strategy like using a default value
                 # or raising an error depending on your data and requirements.
                 # For simplicity here, we'll just print a warning and potentially leave it unencoded
                 # which might cause errors later if the model expects encoded values.
                 # A more robust approach would be to use OneHotEncoder with handle_unknown='ignore'
                 # or map unseen values to a specific category.
                 pass


    # Ensure the order of columns matches the training data
    # This requires knowing the exact order of columns in your training data (X_train or X)
    # You can get this from your Colab notebook: X.columns.tolist()
    # For now, let's assume the order is the same as in the 'features' dictionary
    ordered_columns = features.keys()
    df = df[ordered_columns]


    # Scale the numerical features
    # Identify numerical columns based on the features dictionary
    numerical_features = [f for f, t in features.items() if t == 'number']
    df[numerical_features] = scaler.transform(df[numerical_features])


    return df

# Predict button
if st.sidebar.button("Predict Age"):
    processed_input = preprocess_input(input_data, scaler, label_encoders_dict)

    # Make prediction
    prediction = model.predict(processed_input)

    # Display the prediction
    st.subheader("Prediction:")
    st.success(f"The predicted age is: {prediction[0]:.2f} years")

st.markdown("---")
st.write("This is a simple Streamlit application for age prediction.")
