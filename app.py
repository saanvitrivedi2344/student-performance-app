import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Student Performance Prediction System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('student_performance_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Sidebar inputs
with st.sidebar:
    st.header("📋 Student Details")
    
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_edu = st.selectbox(
        "Parental Education",
        ["some high school", "high school", "some college", 
         "associate's degree", "bachelor's degree", "master's degree"]
    )
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation", ["none", "completed"])
    
    st.markdown("---")
    st.markdown("### 📊 Test Scores")
    math_score = st.slider("Math Score", 0, 100, 65)
    reading_score = st.slider("Reading Score", 0, 100, 65)
    writing_score = st.slider("Writing Score", 0, 100, 65)

# Load models
model, scaler, encoders = load_models()

if model is not None and scaler is not None and encoders is not None:
    
    feature_cols = [
        'gender_encoded', 'race/ethnicity_encoded', 'parental level of education_encoded',
        'lunch_encoded', 'test preparation course_encoded', 'math score',
        'writing score', 'math_reading_diff', 'reading_writing_diff'
    ]
    
    # Prepare input
    student_data = {
        'gender': gender,
        'race/ethnicity': race,
        'parental level of education': parental_edu,
        'lunch': lunch,
        'test preparation course': test_prep,
        'math score': math_score,
        'reading score': reading_score,
        'writing score': writing_score
    }
    
    df_input = pd.DataFrame([student_data])
    
    # Encode categorical variables
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        df_input[col + '_encoded'] = encoders[col].transform(df_input[col])
    
    # Create engineered features
    df_input['math_reading_diff'] = abs(df_input['math score'] - df_input['reading score']) / 100
    df_input['reading_writing_diff'] = abs(df_input['reading score'] - df_input['writing score']) / 100
    
    # Select features
    X_input = pd.DataFrame()
    for col in feature_cols:
        X_input[col] = df_input[col]
    
    # Scale features
    X_scaled = scaler.transform(X_input)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    avg_score = (math_score + reading_score + writing_score) / 3
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Student Profile")
        profile_data = {
            "Attribute": ["Gender", "Race/Ethnicity", "Parental Education", "Lunch", "Test Prep", "Avg Score"],
            "Value": [gender, race, parental_edu, lunch, test_prep, f"{avg_score:.1f}"]
        }
        st.table(pd.DataFrame(profile_data))
    
    with col2:
        st.header("🎯 Prediction Result")
        if prediction == 1:
            st.success("### ✅ PASS")
        else:
            st.error("### ❌ FAIL")
        
        st.metric("Pass Probability", f"{probability[1]:.1%}")
        st.metric("Fail Probability", f"{probability[0]:.1%}")
        
        # Risk assessment
        if probability[1] >= 0.7:
            st.info("🌟 Low Risk Student")
        elif probability[1] >= 0.5:
            st.warning("⚠️ Moderate Risk Student")
        else:
            st.error("🔴 High Risk Student - Intervention Needed")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.markdown("---")
        st.header("🔍 Feature Importance")
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     title='What factors matter most?')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Model files not found. Please upload the .pkl files to the app directory.")

st.markdown("---")
st.markdown("© 2024 Student Performance Prediction | ML Project")