import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. SMART RESOURCE LOADER ---
# Instead of loading a broken pickle, we train a fresh one instantly!
@st.cache_resource
def build_model_from_csv():
    # A. Load Data
    csv_file = 'student_depression_dataset.csv'
    if not os.path.exists(csv_file):
        st.error(f"❌ Error: '{csv_file}' not found. Please put the CSV in the same folder.")
        st.stop()
        
    df = pd.read_csv(csv_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Identify Target and Features
    # We attempt to auto-detect the target column if 'Depression' isn't exact
    possible_targets = ['Depression', 'Depression State', 'Target']
    target_col = next((t for t in possible_targets if t in df.columns), None)
    
    if not target_col:
        st.error(f"❌ Could not find a target column (e.g., 'Depression') in the CSV.")
        st.stop()

    # B. Preprocessing (Encoding)
    encoders = {}
    feature_cols = [col for col in df.columns if col != target_col]
    
    # We work on a copy to avoid SettingWithCopy warnings
    df_train = df.copy()

    for col in feature_cols:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            encoders[col] = le
    
    # Encoding Target if needed
    if df_train[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df_train[target_col] = le_target.fit_transform(df_train[target_col].astype(str))

    # C. Train Model
    X = df_train[feature_cols]
    y = df_train[target_col]
    
    # Create AdaBoost Model (Using default DecisionTree base)
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )
    model.fit(X, y)
    
    return model, encoders, feature_cols, df # Return raw df for min/max values

# Build the model fresh!
model, encoders, feature_cols, df_raw = build_model_from_csv()

# --- 2. UI LAYOUT ---
with st.sidebar:
    # Try to load the image if it exists
    img_path = "Artificial Intelligence Application in Mental Health Research copy.jpg"
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="AI in Mental Health")
    
    st.title("User Inputs")
    st.write("Adjust values below:")

# Main Content
st.title("🧠 Student Depression Risk Predictor")
st.write("This system uses **AdaBoost AI** to assess mental health risks based on your dataset.")
st.divider()

# --- 3. DYNAMIC INPUTS ---
user_inputs = {}

# Create a layout with 2 columns
col1, col2 = st.columns(2)
cols_list = [col1, col2]

for idx, col in enumerate(feature_cols):
    # Toggle between left and right column
    current_col = cols_list[idx % 2]
    
    with current_col:
        # If Categorical (Text) -> Show Dropdown
        if col in encoders:
            options = list(encoders[col].classes_)
            val = st.selectbox(f"{col}", options)
            user_inputs[col] = encoders[col].transform([val])[0]
        
        # If Numerical -> Show Slider
        else:
            min_v = float(df_raw[col].min())
            max_v = float(df_raw[col].max())
            mean_v = float(df_raw[col].mean())
            step = 1.0 if df_raw[col].dtype == 'int64' else 0.1
            
            # Handling case where min == max (single value column)
            if min_v == max_v:
                st.info(f"{col}: {min_v} (Fixed)")
                user_inputs[col] = min_v
            else:
                user_inputs[col] = st.slider(f"{col}", min_v, max_v, mean_v, step)

# --- 4. PREDICTION ---
st.divider()

if st.button("Analyze Risk", type="primary"):
    with st.spinner("Analyzing..."):
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([user_inputs])
        
        # Prediction
        prediction = model.predict(input_data)[0]
        try:
            proba = model.predict_proba(input_data)[0][1] * 100
            confidence = f"{proba:.1f}%"
        except:
            confidence = "N/A"

        # Results
        if prediction == 1:
            st.error("⚠️ Prediction: AT RISK")
            st.write(f"The model suggests a high probability of depression ({confidence}).")
            st.warning("**Recommendation:** Please reach out to a counselor or trusted mentor.")
        else:
            st.success("✅ Prediction: HEALTHY")
            st.write(f"The model suggests a low risk profile ({100 - float(confidence[:-1]):.1f}% healthy).")
            st.balloons()