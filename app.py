import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# --- MODEL BUILDER ---
@st.cache_resource
def build_model_from_csv():
    csv_file = 'student_depression_dataset.csv'

    if not os.path.exists(csv_file):
        st.error("❌ CSV file not found! Upload it to GitHub.")
        st.stop()

    df = pd.read_csv(csv_file)

    # 🔥 CLEANING
    df.columns = df.columns.str.strip()

    # Drop fully empty rows
    df = df.dropna(how='all')

    # ✅ FIXED (new pandas safe method)
    df = df.ffill().bfill()

    # Detect target column
    target_col = None
    for col in df.columns:
        if "depression" in col.lower():
            target_col = col
            break

    if target_col is None:
        st.error("❌ No 'Depression' column found in dataset")
        st.stop()

    encoders = {}

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop any remaining NaN
    df = df.dropna()

    # Split features & target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 🔥 MODEL
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )

    model.fit(X, y)

    return model, encoders, X.columns, df


# Load model
model, encoders, feature_cols, df = build_model_from_csv()

# --- UI ---
st.title("🧠 Student Depression Risk Predictor")
st.write("Fill in the details below to analyze mental health risk.")

user_inputs = {}

col1, col2 = st.columns(2)
cols = [col1, col2]

# Dynamic input fields
for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        if col in encoders:
            val = st.selectbox(col, encoders[col].classes_)
            user_inputs[col] = encoders[col].transform([val])[0]
        else:
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            mean_v = float(df[col].mean())

            if min_v == max_v:
                st.info(f"{col}: {min_v} (fixed)")
                user_inputs[col] = min_v
            else:
                user_inputs[col] = st.slider(col, min_v, max_v, mean_v)

# --- PREDICTION ---
st.divider()

if st.button("Analyze Risk", type="primary"):
    input_df = pd.DataFrame([user_inputs])

    prediction = model.predict(input_df)[0]

    try:
        prob = model.predict_proba(input_df)[0][1] * 100
        confidence = f"{prob:.1f}%"
    except:
        confidence = "N/A"

    if prediction == 1:
        st.error("⚠️ High Risk of Depression")
        st.write(f"Confidence: {confidence}")
        st.warning("👉 Consider talking to a counselor or trusted person.")
    else:
        st.success("✅ Low Risk")
        st.write(f"Confidence: {confidence}")
        st.balloons()
