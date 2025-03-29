import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense # type: ignore
from tensorflow.keras.applications import Xception # type:ignore



st.set_page_config(
    page_title="HEAL AI - Harnessing Machine Learning for Disease Forecasting and Early Detection",
    layout="wide",
    page_icon="🧬"
)

custom_css = """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#f5f5f5, #f5f5f5);
    color: black;
}
.subheader {
    font-size: 20px;
    font-weight: bold;
    color: red;
    background-color: #f5f5f5;
    padding: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
    border-radius: 10px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

st.title("HEAL AI - Harnessing Machine Learning for Disease Forecasting and Early Detection")

working_dir = os.path.dirname(os.path.abspath(__file__))
import joblib

# Load the model
diabetes_model = joblib.load(open(os.path.join(working_dir, 'diabetes_model_final_p2r2.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(working_dir, 'heart_disease_model_p2r2.sav'), 'rb'))


# Load trained model
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os

# For lung cancer
IMAGE_SIZE = (224, 224)
class_labels = ['adenocarcinoma', 'largecellcarcinoma', 'normal', 'squamouscellcarcinoma'] 
lung_cancer_model = load_model(os.path.join(working_dir, 'best_lc.keras'))

# For eye disease
eye_class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
eye_disease_model = load_model(os.path.join(working_dir, 'best_eye.keras'))

def main():
    menu = ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Lung Cancer Prediction", "Eye Disease Prediction"]
    selected = st.sidebar.selectbox("Menu", menu)

    if selected == "Home":
        st.subheader("Home")
        st.write("Welcome to HEAL AI! This application uses machine learning models to predict the likelihood of diseases. Select a disease from the sidebar menu to get started.")
        st.markdown("""
        ### Welcome to the Health Prediction App
        This app helps you predict potential health issues related to diseases like **Diabetes**, **Heart Disease**, **Lung Cancer** and **Eye Disease** based on provided input data or CT image or retinal scans. Please note that these predictions are for informational purposes only and should not be considered a substitute for professional medical advice.

        **How to use:**
        - For each health condition, provide relevant data or upload an image (for diseases like Lung Cancer).
        - Click on the prediction button to see the results.
        - Review the frequently asked questions (FAQs) in the sidebar for more information.
        
        Please ensure you consult a doctor for a definitive diagnosis and appropriate treatment.
    """)

    # Add a disclaimer to the page
        st.markdown("""
        **Disclaimer:** 
        This tool is not intended to diagnose, treat, or provide medical advice. It is important to consult with a healthcare provider for any health concerns.
    """)
        st.markdown("___")

    elif selected == "Diabetes Prediction":
        st.subheader("Diabetes Prediction")
        predict_diabetes(diabetes_model)
        display_diabetes_faq()

    elif selected == "Heart Disease Prediction":
        st.subheader("Heart Disease Prediction")
        predict_heart_disease(heart_disease_model)
        display_heart_disease_faq()

    elif selected == "Lung Cancer Prediction":
        st.subheader("Lung Cancer Prediction")
        predict_lung_cancer(lung_cancer_model)
        display_lung_cancer_faq()
    
    elif selected == "Eye Disease Prediction":
        st.subheader("Eye Disease Prediction")
        predict_eye_disease(eye_disease_model)
        display_eye_disease_faq()

def predict_diabetes(model):
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL)')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value (mmHg)')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value (in mm)')

    with col2:
        Insulin = st.text_input('Insulin Level (IU/ml)')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person (in years)')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic.'
        else:
            diab_diagnosis = 'The person is not diabetic.'

        st.subheader("Diabetes Prediction Result:")
        st.success(diab_diagnosis)

def predict_heart_disease(model):
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age (in years)')

    with col2:
        Sex = st.selectbox('Sex', ['Male', 'Female'])

    with col3:
        CP = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])

    with col1:
        Trestbps = st.text_input('Resting Blood Pressure (mmHg)')

    with col2:
        Chol = st.text_input('Serum Cholestoral (mg/dL)')

    with col3:
        FBS = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['Yes', 'No'])

    with col1:
        Restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])

    with col2:
        Thalach = st.text_input('Maximum Heart Rate Achieved')

    with col3:
        Exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])

    with col1:
        Oldpeak = st.text_input('ST Depression Induced by Exercise')

    with col2:
        Slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

    with col3:
        CA = st.selectbox('Number of Major Vessels (0-3)', ['0', '1', '2', '3'])

    with col1:
        Thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        user_input = [Age, Sex, CP, Trestbps, Chol, FBS, Restecg, Thalach, Exang, Oldpeak, Slope, CA, Thal]
        user_input = [float(x) if x.replace('.', '', 1).isdigit() else 0 for x in user_input]
        heart_prediction = model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person has heart disease.'
        else:
            heart_diagnosis = 'The person does not have heart disease.'

        st.subheader("Heart Disease Prediction Result:")
        st.success(heart_diagnosis)

def predict_lung_cancer(model):
    uploaded_file = st.file_uploader("Upload a chest CT scan image file for Lung Cancer prediction", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Load image and preprocess
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        if st.button('Lung CT Scan Test Result'):
            if predicted_label == class_labels[0]:
                lung_diagnosis = 'The person seems to have adenocarcinoma type of lung cancer. Please consult an oncologist.'
            elif predicted_label == class_labels[1]:
                lung_diagnosis = 'The person seems to have large cell carcinoma type of lung cancer. Please consult an oncologist.'
            elif predicted_label == class_labels[2]:
                lung_diagnosis = 'The person does not seem to have any type of lung cancer.'
            elif predicted_label == class_labels[3]:
                lung_diagnosis = 'The person seems to have squamous cell carcinoma type of lung cancer. Please consult an oncologist.'

            st.subheader("CT Scan Result:")
            st.success(lung_diagnosis)

        #Debug: Display raw predictions
    #st.write(f"### **Confidence Scores:** {predictions[0]}")
        #st.write(f"### **Predicted Class: {predicted_label}**")

from io import BytesIO
def predict_eye_disease(model):
    uploaded_file = st.file_uploader("Upload a retinal image for Eye Disease prediction", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Load image and preprocess
        img = image.load_img(uploaded_file, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = eye_class_labels[predicted_class]

        if st.button('Eye Disease Result'):
            if predicted_label == eye_class_labels[0]:
                eye_diagnosis = 'The person seems to have cataract. Please consult an opthalmologist.'
            elif predicted_label == eye_class_labels[1]:
                eye_diagnosis = 'The person seems to have diabetic retinopathy. Please consult an opthalmologist.'
            elif predicted_label == eye_class_labels[2]:
                eye_diagnosis = 'The person seems to have glaucoma. Please consult an opthalmologist.'
            elif predicted_label == eye_class_labels[3]:
                eye_diagnosis = 'The person seems to have healthy eyes.'

            st.subheader("Result:")
            st.success(eye_diagnosis)

def display_faq(faqs):
    for question, answer in faqs.items():
        with st.sidebar.expander(question):
            st.write(answer)

def display_diabetes_faq():
    with st.sidebar.expander("Diabetes FAQ"):
        diabetes_faqs = {
            "Diabetes FAQ":"FAQs below regarding diabetes.",
            "What is diabetes?": "Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.",
            "What are the symptoms of diabetes?": "Symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, extreme hunger, fatigue, blurred vision, and slow healing of wounds.",
            "What values should I input for Insulin Level?": "Enter the insulin level given on your report in IU/ml (International Units per milliliter).",
            "What values should I input for BMI?": "Enter the BMI value. BMI is calculated as weight (kg) divided by height squared (m^2).",
            "What values should I input for Diabetes Pedigree Function(DPF)?": "DPF = Σ(0.1 × history), where Σ represents summation, and history represents the degree of relationship for each relative with diabetes(coded 1 for 1st-degree relatives, 0.5 for 2nd-degree relatives, and 0.25 for 3rd-degree relatives).",
            "What should I do in case of a diabetic emergency?": "In case of a diabetic emergency such as hypoglycemia (low blood sugar) or hyperglycemia (high blood sugar), it is important to take appropriate action. For hypoglycemia, consume sugary foods or drinks to raise blood sugar levels. For hyperglycemia, monitor blood sugar levels closely, drink plenty of water, and seek medical attention (call 108 or 112 (India)) .",
            "What are the signs of a diabetic emergency?": "Signs of a diabetic emergency include dizziness, confusion, sweating, trembling, weakness, extreme thirst, frequent urination, nausea, vomiting, abdominal pain, fruity-smelling breath (in case of hyperglycemia), and unconsciousness.",
        }

        display_faq(diabetes_faqs)

def display_heart_disease_faq():
    with st.sidebar.expander("Heart Disease FAQ"):
        heart_disease_faqs = {
            "Heart Disease FAQs":"FAQs below regarding heart diseases.",
            "What are the risk factors for heart disease?": "Risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, poor diet, physical inactivity, and excessive alcohol consumption.",
            "How is heart disease diagnosed?": "Heart disease is diagnosed through various tests such as electrocardiogram (ECG), echocardiogram, stress tests, blood tests, and coronary angiography.",
            "What values should I input for Serum Cholestoral?": "Enter the total serum cholestoral value in mg/dL (in your lipid profile report).",
            "What values should I input for Resting Electrocardiographic Results?": "It is written at top of your ECG (the wavy one :)) report (put normal if Normal Sinus Rythm is written).",
            "What values should I input for ST depression induced by exercise?": "When entering this value, provide the magnitude of ST segment depression (in mm) observed during exercise testing. If no ST segment depression is observed, enter 0.",
            "What values should I input for Number of Major Vessels Colored by Flourosopy?": "When entering this value, provide the number of major vessels (0-3) that appear to have significant narrowing or blockages as observed during the cardiac examination.",
            "What values should I input for Thal?": "Thal refers to thallium stress testing, which is a type of nuclear imaging test used to evaluate blood flow to the heart muscle.(If not done, put normal as its rare.)",
            "What are the signs of a heart attack?": "Common signs of a heart attack include chest discomfort or pain (often described as pressure, tightness, or squeezing), shortness of breath, nausea or vomiting, lightheadedness or dizziness, discomfort in the arms, back, neck, jaw, or stomach, and cold sweats. Not everyone experiences these symptoms, and they can vary between individuals.",
            "What should I do during a heart attack?": "During a heart attack, it's crucial to act quickly. Call emergency services immediately (108 or 112 (India) or your local emergency number). While waiting for help, chew and swallow aspirin (if available) to help reduce blood clotting. Stay calm, and if you have been prescribed nitroglycerin, take it as directed. If you are experiencing chest pain, sit down and rest in a comfortable position. Avoid exertion or stress.",
            "What should I do if someone is having a heart attack?": "If you suspect someone is having a heart attack, call emergency services immediately (108 or 112 (India) or your local emergency number). Encourage the person to sit down and rest in a comfortable position. If they have been prescribed nitroglycerin/sorbitol, assist them in taking it as directed. Stay with the person and monitor their condition while waiting for help to arrive.",
            # Add more FAQs specific to heart disease
        }

        display_faq(heart_disease_faqs)

def display_lung_cancer_faq():
    with st.sidebar.expander("Lung Cancer FAQ"):
        lung_cancer_faqs = {
            "Lung Cancer FAQs":"FAQs below regarding lung cancer.",
            "What is lung cancer?": "Lung cancer is a type of cancer that begins in the lungs. It is typically caused by smoking, but non-smokers can also develop lung cancer. Symptoms may include persistent coughing, coughing up blood, shortness of breath, chest pain, and weight loss.",
            "What are the risk factors for lung cancer?": "The main risk factor for lung cancer is smoking. However, exposure to secondhand smoke, radon, asbestos, and other environmental factors can also increase the risk of developing lung cancer.",
            "How is lung cancer diagnosed?": "Lung cancer is diagnosed through imaging tests such as chest X-rays, CT scans, and PET scans, as well as biopsy procedures to confirm the presence of cancer cells.",
            "What are the symptoms of lung cancer?": "Symptoms include persistent coughing, coughing up blood, shortness of breath, chest pain, and weight loss. Some people may not experience symptoms until the cancer is advanced.",
            "What treatments are available for lung cancer?": "Treatment options for lung cancer include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy. The treatment plan depends on the stage and type of lung cancer.",
        }
        display_faq(lung_cancer_faqs)

def display_eye_disease_faq():
    with st.sidebar.expander("Eye Disease FAQ"):
        eye_disease_faqs = {
            "Eye Disease FAQs": "FAQs below regarding common eye diseases.",
            
            "What is an eye disease?": "Eye disease refers to any condition that affects the eyes, potentially leading to vision impairment or blindness. Common eye diseases include cataract, diabetic retinopathy, and glaucoma.",
            
            "What are the causes of eye diseases?": "Causes of eye diseases vary and may include aging, diabetes, high blood pressure, genetics, infections, injuries, and prolonged exposure to UV light or screens.",
            
            "What are the types of eye diseases?": 
                "- **Cataract:** A clouding of the eye's lens, leading to blurry vision.\n"
                "- **Diabetic Retinopathy:** Damage to the blood vessels in the retina caused by diabetes, which can lead to vision loss.\n"
                "- **Glaucoma:** A condition that damages the optic nerve, often due to high intraocular pressure, which can cause blindness if untreated.",
            
            "How are eye diseases diagnosed?": "Eye diseases are diagnosed through comprehensive eye exams, including visual acuity tests, tonometry (to measure eye pressure), retinal imaging, and OCT scans.",
            
            "What are the treatment options for eye diseases?": 
                "- **Cataract:** Surgery to replace the clouded lens with an artificial one.\n"
                "- **Diabetic Retinopathy:** Laser therapy, medications (anti-VEGF), or surgery.\n"
                "- **Glaucoma:** Eye drops, oral medications, laser therapy, or surgery to reduce eye pressure.",
            
            "How can I care for my eyes and prevent diseases?": 
                "- Get regular eye exams.\n"
                "- Wear sunglasses with UV protection.\n"
                "- Follow a healthy diet rich in vitamins A, C, and E.\n"
                "- Maintain good blood sugar and blood pressure levels.\n"
                "- Take breaks from screens to reduce eye strain.",
        }
        display_faq(eye_disease_faqs)


def display_faq(faqs):
    for question, answer in faqs.items():
        with st.sidebar.expander(question):
            st.write(answer)


if __name__ == "__main__":
    main()
