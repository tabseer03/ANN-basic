import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


# Load the trained model
model = tf.keras.models.load_model('model.keras')


# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('ohe_geography.pkl', 'rb') as f:
    ohe_geography = pickle.load(f)


# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography=st.selectbox("Geography", ["France", "Spain", "Germany"])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider("Age", 18, 100)
tenure=st.slider("Tenure", 0, 10)
balance=st.number_input("Balance")
creditscore=st.slider("Credit Score")
estimatedsalary=st.number_input("Estimated Salary")
has_credit_card=st.selectbox("Has Credit Card", [0, 1])
is_active_member=st.selectbox("Is Active Member", [0, 1])
num_of_products=st.slider("Number of Products", 1, 4)

input_data=pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure], 
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimatedsalary]
})

geo_encoded=ohe_geography.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geography.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_data)


#prediction churn

prediction=model.predict(input_data_scaled)
prediction_probability=prediction[0][0]

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
