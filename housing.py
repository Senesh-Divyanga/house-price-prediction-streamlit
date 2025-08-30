
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

data = { 'area': [1000, 1500, 2000, 2500, 3000], 'bedrooms': [2, 3, 3, 4, 4], 'age': [10, 15, 20, 5, 8], 'price': [100000, 150000, 200000, 250000, 270000]}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'age']]

y = df['price']

model = LinearRegression()

model.fit(X, y)

joblib.dump(model, 'model.pkl')

import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title(" House Price Prediction App")

st.write("Enter house details to predict the price:")


area = st.number_input("Area (sq ft)", value=1000)
bedrooms = st.number_input("Number of Bedrooms", value=2, step=1)
age = st.number_input("Age of House (years)", value=10)

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, age]])
    prediction = model.predict(features)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")