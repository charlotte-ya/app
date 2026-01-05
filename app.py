import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="CVD Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Prediction System"])
st.sidebar.markdown("---")
st.sidebar.info("WQD7001 Group 9 Project\n")

# --- Page 1: Home ---
if app_mode == "Home":
    st.title("Cardiovascular Disease Risk Assessment")
    
    # 1. Introduction
    st.markdown("""
    ### 👋 Welcome
    This application utilizes a **Random Forest** machine learning model to predict the **10-year risk** of developing Coronary Heart Disease (CHD). 
    
    #### 🚀 How to use
    1. Go to the **Prediction System** tab.
    2. Enter your health details (Demographics, Medical History, Clinical Measurements).
    3. Click **Run Analysis** to see your risk profile.
    """)
    
    st.divider()
    
    # 2. Standards with Corrected Links
    st.markdown("### 🏥 Medical Standards Reference")
    st.markdown("Your health metrics are compared against the following international guidelines:")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info("**Blood Pressure**")
        st.markdown("""
        **Target: < 120/80 mmHg**
        *Values 130-139(Systolic), 80-89 (Diastolic) are considered Stage 1 Hypertension.*
        
        Source: [AHA - BP Understanding](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings)
        """)
        
    with c2:
        st.info("**Glucose Level**")
        st.markdown("""
        **Target: < 100 mg/dL**
        *100-125 mg/dL is Prediabetes.*
        
        Source: [ADA - Diagnosis Standards](https://diabetes.org/about-diabetes/diagnosis)
        """)
        
    with c3:
        st.info("**Body Mass Index (BMI)**")
        st.markdown("""
        **Target: < 25.0 kg/m²**
        *≥ 25.0 is Overweight.*
        
        Source: [WHO - Obesity Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)
        """)

    st.caption("Note: These baselines are for screening purposes. Please consult a doctor for a formal diagnosis.")

# --- Page 2: Prediction System ---
elif app_mode == "Prediction System":
    st.title("Risk Prediction Tool")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("Patient Data Input")

    def user_input_features():
        # Group 1: Demographics
        st.sidebar.subheader("1. Demographics")
        gender = st.sidebar.radio("Gender", ('Male', 'Female'))
        age = st.sidebar.slider('Age', 30, 80, 50)
        education = st.sidebar.selectbox("Education Level", (1, 2, 3, 4), help="1: Some High School, 4: College/Degree")

        # Group 2: Habits
        st.sidebar.subheader("2. Behavioral Habits")
        currentSmoker = st.sidebar.radio("Current Smoker?", ('No', 'Yes'))
        
        cigsPerDay = 0.0
        if currentSmoker == 'Yes':
            cigsPerDay = st.sidebar.number_input('Cigarettes Per Day', min_value=1, max_value=70, value=10, step=1)
        # Group 3: Medical History
        st.sidebar.subheader("3. Medical History")
        bp_meds = st.sidebar.checkbox("On BP Medication?")
        prevalentStroke = st.sidebar.checkbox("History of Stroke?")
        prevalentHyp = st.sidebar.checkbox("Hypertension (High BP)?")
        diabetes = st.sidebar.checkbox("Diabetes?")

        # Group 4: Measurements
        st.sidebar.subheader("4. Clinical Measurements")
        sysBP = st.sidebar.slider('Systolic BP (sysBP)', 80, 250, 120, help="Normal < 120 mmHg")
        # Clarified the help text here
        diaBP = st.sidebar.slider('Diastolic BP (diaBP)', 40, 150, 80, help="Normal < 80 mmHg (80+ is elevated)")
        bmi = st.sidebar.number_input('BMI', 10.0, 50.0, 25.0, step=0.1)
        heartRate = st.sidebar.slider('Heart Rate', 40, 150, 75)
        glucose = st.sidebar.slider('Glucose Level', 40, 400, 85, help="Normal < 100 mg/dL")
        totChol = st.sidebar.slider('Total Cholesterol', 100, 600, 200)

        data = {
            'male': 1 if gender == 'Male' else 0,
            'age': age,
            'education': education,
            'currentSmoker': 1 if currentSmoker == 'Yes' else 0,
            'cigsPerDay': cigsPerDay,
            'BPMeds': 1 if bp_meds else 0,
            'prevalentStroke': 1 if prevalentStroke else 0,
            'prevalentHyp': 1 if prevalentHyp else 0,
            'diabetes': 1 if diabetes else 0,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': bmi,
            'heartRate': heartRate,
            'glucose': glucose
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    with st.expander("View Input Data Summary"):
        st.dataframe(input_df)

    st.divider()

    # --- Prediction Execution ---
    if st.button('Run Analysis', type="primary", use_container_width=True):
        
        with st.spinner('Analyzing...'):
            prediction_proba = model.predict_proba(input_df)
            risk_prob = prediction_proba[0][1] 
        
        # --- Logic to determine Label ---
            if risk_prob > 0.5:
                risk_label = "High Risk"
            else:
                risk_label = "Low Risk"

        # --- 1. Risk Assessment Results ---
        st.subheader("1. Risk Assessment Results")
        st.metric(
            # label="Prediction Outcome", 
            label = '',
            value=risk_label
        )
        
        st.caption(f"The calculated probability is **{risk_prob:.1%}** (Threshold: 50%).")
        
        if risk_prob > 0.5:
            st.error(f"**Analysis**: The model identifies this profile as **High Risk** because the probability ({risk_prob:.1%}) exceeds the safety threshold.")
        else:
            st.success(f"**Analysis**: The model identifies this profile as **Low Risk** because the probability ({risk_prob:.1%}) is within the safe range.")

        st.markdown("---")

        # --- 2. Vitals Analysis ---
        st.subheader("2. Vitals Analysis")
        st.markdown("**Comparison against standard clinical thresholds:**")
        
        k1, k2, k3, k4 = st.columns(4)
        
        # Values logic
        u_sys = input_df['sysBP'][0]
        u_dia = input_df['diaBP'][0]
        u_glu = input_df['glucose'][0]
        u_bmi = input_df['BMI'][0]

        # Metrics with Delta
        # Logic: If delta is 0, it means it's exactly ON the threshold, which usually needs attention (e.g. Dia 80 is Stage 1)
        # But for simplicity, we show the math difference.
        
        k1.metric("Systolic BP", f"{u_sys}", delta=f"{u_sys-120} (Ref: <120)", delta_color="inverse")
        k2.metric("Diastolic BP", f"{u_dia}", delta=f"{u_dia-80} (Ref: <80)", delta_color="inverse")
        k3.metric("Glucose", f"{u_glu}", delta=f"{u_glu-100} (Ref: <100)", delta_color="inverse")
        k4.metric("BMI", f"{u_bmi:.1f}", delta=f"{u_bmi-25:.1f} (Ref: <25.0)", delta_color="inverse")
        
        st.caption("Comparison logic based on AHA, ADA, and WHO guidelines mentioned on the Home page.")

    st.markdown("---")
    st.caption("END")