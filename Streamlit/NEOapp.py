
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from joblib import dump, load




pipiline_path = '/Volumes/Maestria/Maestria/Programs/Machine Learning/Proyecto/Exports/pipeline.sav'

with open(pipiline_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline = joblib.load(pipiline_path)
    print("pipeline loaded successfully!")
except Exception as e:
    print("Failed to load pipeline:", e)

model_path_log = '/Volumes/Maestria/Maestria/Programs/Machine Learning/Proyecto/Exports/best_model_log.sav'

with open(model_path_log, 'rb') as file:
    print(file.read(100))  
try:
    model_log = joblib.load(model_path_log)
    print("Logistic regression model loaded successfully!")
except Exception as e:
    print("Failed to load logistic regression model:", e)

model_path_svm = '/Volumes/Maestria/Maestria/Programs/Machine Learning/Proyecto/Exports/svm.sav'

with open(model_path_svm, 'rb') as file:
    print(file.read(100))  
try:
    model_svm = joblib.load(model_path_svm)
    print("Support vector machine model loaded successfully!")
except Exception as e:
    print("Failed to load support vector machine model:", e)


#print(pipeline)

st.set_page_config(page_title='Near Earth Objects hazard prediction', layout='wide')


st.title('Near Earth Objects hazard prediction 🌠')
st.header('Input Features')

col1, col2 = st.columns(2)


with col1:
    name = st.text_input('Enter a name')
    absolute_magnitude = st.number_input('Enter a absolute magnitude of its intrinsic luminosity(9 to 34)', min_value= 9.0, max_value=34.0, value=10.0, step=1.0)
    estimated_diameter_min = st.number_input('Enter a minimum estimated diameter in kilometers (0.001 to 40)', min_value=0.001, max_value=40.0, value=5.0, step=1.0, format= '%.000f')

with col2:
    estimated_diameter_max = st.number_input('Enter a maximum estimated diameter in kilometers (0.001 to 90)', min_value=0.001, max_value=90.0, value=5.0, step=1.0, format= '%.000f')
    relative_velocity = st.number_input('Enter its velocity relative to Earth in km/h (200 to 300,000)', min_value=200.0, max_value=300000.0, value=500.0, step=1.0, format= '%.0f')
    miss_distance = st.number_input('Enter a miss distance relative to Earth in kilometers (6000 to 75,000,000)', min_value=6000.0, max_value=75000000.0, value=10000.0, step=1.0, format= '%.0f')


if st.button('Predict'):
    input_data = pd.DataFrame(
    
        {'absolute_magnitude': [absolute_magnitude], 'estimated_diameter_min': [estimated_diameter_min],
         'estimated_diameter_max': [estimated_diameter_max], 'relative_velocity': [relative_velocity], 'miss_distance': [miss_distance]},
         index=[0]
    )

    st.write(input_data)

    pipelined_data = pipeline.transform(input_data)

    col3, col4 = st.columns(2)

    with col3:
        predictionlog  = model_log.predict(pipelined_data)
    
    if predictionlog[0] == 1:
        st.write('The predicted hazard for ', name, ' is HIGH')
    else:
        st.write('The predicted hazard for ', name, ' is LOW')
