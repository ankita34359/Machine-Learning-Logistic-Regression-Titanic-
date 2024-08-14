import joblib
import streamlit as st
import pandas as pd
import numpy as np

Titanic_model_pkl = r"C:\Users\Lenovo\OneDrive\Desktop\Juypter_projects\Titanic_model.pkl"
loaded_model = joblib.load(Titanic_model_pkl)

st.header("Titanic Passenger Survival Prediction Model")

Passenger_class=st.number_input("Enter the Passenger Class")

sex=st.selectbox("Enter the Sex ", ("male", "female"))
sex_dict={"male":0, "female":1}
sex=sex_dict[sex]

age=st.number_input("Enter the Age ")

SibSp = st.number_input("Enter the number of siblings or spouse aboard")

Parch = st.number_input("Enter the number of parents or children aboard")

Fare = st.number_input("Enter the amount of fare")

Embarked=st.selectbox("Enter the Embarkation ", ("Cherbourg", "Queenstown", "Southampton"))
Embarked_dict={"Cherbourg":0, "Queenstown":1, "Southampton":2}
Embarked=Embarked_dict[Embarked]

# Prepare the input array
X_new = np.array([[Passenger_class, sex, age, SibSp, Parch, Fare, Embarked]])

# Prediction
if st.button("Predict"):
    prediction = loaded_model.predict(X_new)
    result = "Survived" if prediction == 1 else "not survive"
    st.write(f"The person would have {result}")
