import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Depression Predictor", layout="wide")

# Load model instantly
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model, encoders, feature_cols = load_model()

st.title("🧠 Student Depression Predictor")

user_inputs = {}

col1, col2 = st.columns(2)
cols = [col1, col2]

for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        if col in encoders:
            val = st.selectbox(col, encoders[col].classes_)
            user_inputs[col] = encoders[col].transform([val])[0]
        else:
            user_inputs[col] = st.number_input(col)

if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")
