import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load("model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
embarked_encoder = joblib.load("embarked_encoder.pkl")

# Title and description
st.title("Titanic Survival Prediction App 🚢")
st.write("Enter passenger details to predict survival:")

# Input fields
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("No. of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("No. of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Prediction
if st.button("Predict"):
    sex_val = sex_encoder.transform([sex])[0]
    embarked_val = embarked_encoder.transform([embarked])[0]

    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_val,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_val
    }])

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have survived.")
    else:
        st.error("💀 The passenger would not have survived.")
