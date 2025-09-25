import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('best_model.pkl')

st.title("Student Performance Predictor")

study_time = st.slider("Study Time (hours per day)", 0, 12, 2)
social_media_time = st.slider("Social Media Time (hours per day)", 0, 12, 2)
attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
mental_health = st.slider("Mental Health Score (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours (per night)", 0, 12, 7)
part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Performance"):
    input_data= np.array([[study_time, social_media_time, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]
    prediction =  max(0, min(100, prediction))
    st.success(f"Predicted Performance Score: {prediction:.2f}/100")
    
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts student performance using a trained ML model. \n\n"
    "Adjust the sliders and select options, then click 'Predict Performance'. \n\n"
    "Built with Streamlit, NumPy, and Scikit-learn."
)


st.markdown("---")  # horizontal line
st.markdown("© 2025 Lincoln Chitswa | For educational purposes only")
