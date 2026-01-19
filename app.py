import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open('diabetes_svc_model.pkl','rb') as f:
    model = pickle.load(f)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) :
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    )
    prediction = model.predict(input_data)[0]
    
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    probability = model.predict_proba(input_data)[0][1] 
    return f"{result} (Probability: {probability:.2f})"

inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="Blood Pressure", value=70),
    gr.Number(label="Skin Thickness", value=20),
    gr.Number(label="Insulin", value=79),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="Diabetes Pedigree Function", value=0.5),
    gr.Number(label="Age", value=30)
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title='Diabetes Prediction'
)    

app.launch()