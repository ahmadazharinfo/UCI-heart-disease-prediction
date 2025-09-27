import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved objects
scaler = joblib.load(r"D:\Learning\AI & Machine Learning Engineering\Sprints x Microsoft\model\scaler.pkl")
pca = joblib.load(r"D:\Learning\AI & Machine Learning Engineering\Sprints x Microsoft\model\pca.pkl")
model = joblib.load(r"D:\Learning\AI & Machine Learning Engineering\Sprints x Microsoft\model\best_heart_disease_model.pkl")

selected_pca_indices = [12, 4, 8, 1, 11, 10, 14, 0]

st.title("🫀 Heart Disease Prediction")
st.header("Enter Patient Data:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("Sex", ["Female", "Male"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    exang = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])

with col2:
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0, step=1)
    thal = st.selectbox("Thalassemia Type", ["Other", "6", "7"])
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    slope = st.selectbox("ST Slope", [1, 2, 3])

if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    try:
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "Yes" else 0
        exang_val = 1 if exang == "Yes" else 0
        
        thal_6 = 1 if thal == "6" else 0
        thal_7 = 1 if thal == "7" else 0
        
        cp_2 = 1 if cp == 2 else 0
        cp_3 = 1 if cp == 3 else 0
        cp_4 = 1 if cp == 4 else 0
        
        restecg_1 = 1 if restecg == 1 else 0
        restecg_2 = 1 if restecg == 2 else 0
        
        slope_2 = 1 if slope == 2 else 0
        slope_3 = 1 if slope == 3 else 0
        
        features = np.array([[
            age, sex_val, trestbps, chol, fbs_val, thalach, exang_val,
            oldpeak, ca, thal_6, thal_7, cp_2, cp_3, cp_4,
            restecg_1, restecg_2, slope_2, slope_3
        ]])
        
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        features_selected = features_pca[:, selected_pca_indices]
        
        prediction = model.predict(features_selected)
        prediction_proba = model.predict_proba(features_selected)
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if len(prediction_proba[0]) == 2:
                disease_prob = prediction_proba[0][1]
                if disease_prob >= 0.7:
                    st.error("🔴 **High Risk - Heart Disease Detected**")
                elif disease_prob >= 0.4:
                    st.warning("🟡 **Moderate Risk**")
                else:
                    st.success("✅ **Low Risk - No Heart Disease**")
                st.metric("Heart Disease Probability", f"{disease_prob*100:.1f}%")
            elif len(prediction_proba[0]) == 5:
                max_class = np.argmax(prediction_proba[0])
                if max_class >= 3:
                    st.error(f"🔴 **High Risk - Class {max_class}**")
                elif max_class >= 1:
                    st.warning(f"🟡 **Moderate Risk - Class {max_class}**")
                else:
                    st.success(f"✅ **Low Risk - Class {max_class}**")
                st.metric("Primary Class Probability", f"{prediction_proba[0][max_class]*100:.1f}%")
        
        with result_col2:
            if len(prediction_proba[0]) == 2:
                st.write(f"🔴 Disease Risk: {prediction_proba[0][1]*100:.1f}%")
                st.write(f"🟢 No Disease: {prediction_proba[0][0]*100:.1f}%")
            else:
                for i, prob in enumerate(prediction_proba[0]):
                    risk = ["Very Low", "Low", "Moderate", "High", "Very High"][i]
                    st.write(f"Class {i} ({risk}): {prob*100:.1f}%")
        
        # FIXED: Medical Interpretation section moved outside classification blocks
        st.subheader("🩺 Medical Interpretation")
        
        # Handle binary classification
        if len(prediction_proba[0]) == 2:
            disease_prob = prediction_proba[0][1]
            if disease_prob >= 0.8:
                st.error("🔴 **Very High Risk:** Immediate medical attention recommended.")
            elif disease_prob >= 0.6:
                st.warning("🟡 **High Risk:** Schedule appointment within days.")
            elif disease_prob >= 0.4:
                st.warning("🟠 **Moderate Risk:** Consider medical consultation.")
            elif disease_prob >= 0.2:
                st.info("🟡 **Low-Moderate Risk:** Continue monitoring.")
            else:
                st.success("🟢 **Low Risk:** Maintain healthy lifestyle.")
            
            st.progress(disease_prob)
        
        # Handle multi-class classification
        else:
            max_class = np.argmax(prediction_proba[0])
            max_prob = prediction_proba[0][max_class]
            
            if max_class == 0:
                st.success("🟢 **No Heart Disease:** Maintain healthy lifestyle.")
            elif max_class == 1:
                st.info("🟡 **Mild Risk:** Continue monitoring and healthy habits.")
            elif max_class == 2:
                st.warning("🟠 **Moderate Risk:** Consider medical consultation and lifestyle improvements.")
            elif max_class == 3:
                st.warning("🟡 **High Risk:** Schedule appointment within days.")
            else:
                st.error("🔴 **Very High Risk:** Immediate medical attention recommended.")
            
            st.progress(max_prob)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("""
**⚠️ Medical Disclaimer:** 
This tool is for educational purposes only and should not replace professional medical advice.
Always consult with a qualified healthcare provider for medical concerns.
""")

with st.sidebar:
    st.header("📋 Feature Explanations")
    
    st.subheader("👤 Demographics")
    st.write("**Age:** Patient age in years (20-100)")
    st.write("**Sex:** Male (1) or Female (0)")
    
    st.subheader("🩺 Vital Signs")
    st.write("**Resting BP:** Blood pressure at rest (80-200 mmHg)")
    st.write("**Cholesterol:** Serum cholesterol (100-600 mg/dl)")
    st.write("**Max Heart Rate:** Highest heart rate achieved (60-220 bpm)")
    
    st.subheader("🔬 Clinical Tests")
    st.write("**Fasting Blood Sugar:** >120 mg/dl indicates diabetes risk")
    st.write("**Resting ECG:** 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy")
    st.write("**Oldpeak:** ST depression from exercise (0-10)")
    st.write("**ST Slope:** 1=Upsloping, 2=Flat, 3=Downsloping")
    
    st.subheader("💔 Cardiac Indicators")
    st.write("**Exercise Angina:** Chest pain during exercise (Yes/No)")
    st.write("**Chest Pain Type:** 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic")
    st.write("**Major Vessels:** Number of vessels colored by fluoroscopy (0-4)")
    st.write("**Thalassemia:** 6=Fixed defect, 7=Reversible defect, Other=Normal")
