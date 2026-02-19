import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary Prediction App")

age = st.number_input("Age", min_value=18, max_value=60)

gender = st.selectbox("Gender", encoder["Gender"].classes_)
education_level = st.selectbox("Education Level", encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title", encoder["Job Title"].classes_)
years_of_exp = st.number_input("Years of Experience", 0, 40)

df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education_level],
    "Job Title": [job_title],
    "Years of Experience": [years_of_exp]
})

if st.button("Prediction"):

    # Encode categorical columns
    for col in encoder.keys():
        df[col] = encoder[col].transform(df[col])

    # Predict
    prediction = model.predict(df)
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
