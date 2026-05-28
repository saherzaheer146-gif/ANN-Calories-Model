import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

st.title('ANN Calories Prediction Model')

# Model training ka code yahan
dataset = pd.read_csv('calories.csv')
X = dataset.drop('Calories', axis=1)
y = dataset['Calories']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=X_scaled.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=50, verbose=0)

st.success('Model trained!')

# User input
st.sidebar.header('Input Data')
age = st.sidebar.number_input('Age', 10, 100)
weight = st.sidebar.number_input('Weight', 30, 200)
height = st.sidebar.number_input('Height', 100, 250)

if st.button('Predict Calories'):
    input_data = scaler.transform([[age, weight, height]])
    prediction = model.predict(input_data)
    st.write(f'Predicted Calories: {prediction[0][0]:.2f}')
