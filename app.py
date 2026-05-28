import streamlit as st
import pandas as pd
import numpy as np
# import tensorflow as tf  # Comment kar diya
# model = tf.keras.models.load_model('model.h5')  # Comment kar diya

st.set_page_config(page_title="Calories Predictor", layout="centered")

st.title("🔥 Calories Burn Predictor")
st.write("Bas details dalo, calories estimate ho jayegi")

# Input fields
age = st.number_input("Age", 10, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight kg", 30.0, 200.0, 70.0)
height = st.number_input("Height cm", 100.0, 220.0, 170.0)
duration = st.number_input("Duration minutes", 1, 180, 30)
heart_rate = st.number_input("Heart Rate", 60, 200, 120)
body_temp = st.number_input("Body Temp °C", 35.0, 42.0, 37.0)

if st.button("Predict Calories"):
    # Dummy formula - tensorflow ke bina
    calories = (weight * duration * 0.1) + (heart_rate * 0.2) - (age * 0.5)
    if gender == "Male":
        calories += 50
    
    st.success(f"🔥 Estimated Calories Burned: {calories:.0f} kcal")
    st.info("Note: Ye dummy estimate hai. Model subha add kar denge")

st.write("---")
st.caption("ANN Calories Model - Streamlit App")
