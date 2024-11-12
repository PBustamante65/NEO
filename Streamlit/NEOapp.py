
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
sys.path.append('/Volumes/Maestria/GitHub/NEO/NEO/Exports/API')

import dataprocess as dp

# pipiline_path = 'Exports/pipeline.sav'

# with open(pipiline_path, 'rb') as file1:
#     print(file1.read(100))  
# try:
#     pipeline = joblib.load(pipiline_path)
#     print("pipeline loaded successfully!")
# except Exception as e:
#     print("Failed to load pipeline:", e)

# model_path_log = 'Exports/best_model_log.sav'

# with open(model_path_log, 'rb') as file:
#     print(file.read(100))  
# try:
#     model_log = joblib.load(model_path_log)
#     print("Logistic regression model loaded successfully!")
# except Exception as e:
#     print("Failed to load logistic regression model:", e)

# model_path_svm = 'Exports/svm.sav'

# with open(model_path_svm, 'rb') as file:
#     print(file.read(100))  
# try:
#     model_svm = joblib.load(model_path_svm)
#     print("Support vector machine model loaded successfully!")
# except Exception as e:
#     print("Failed to load support vector machine model:", e)


pipeline5_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/pipeline5.sav'

with open(pipeline5_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline5 = joblib.load(pipeline5_path)
    print("pipeline 5 loaded successfully!")
except Exception as e:
    print("Failed to load pipeline 5:", e)

pipeline11_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/pipeline11.sav'

with open(pipeline11_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline11 = joblib.load(pipeline11_path)
    print("pipeline 11 loaded successfully!")
except Exception as e:
    print("Failed to load pipeline 11:", e)

pipeline15_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/pipeline15.sav'

with open(pipeline15_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline15 = joblib.load(pipeline15_path)
    print("pipeline 15 loaded successfully!")
except Exception as e:
    print("Failed to load pipeline 15:", e)

pipeline18_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/pipeline18.sav'

with open(pipeline18_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline18 = joblib.load(pipeline18_path)
    print("pipeline 18 loaded successfully!")
except Exception as e:
    print("Failed to load pipeline 18:", e)

model5_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/best_model_5.sav'

with open(model5_path, 'rb') as file:
    print(file.read(100))
try:
    model5 = joblib.load(model5_path)
    print("model 5 loaded successfully!")
except Exception as e:
    print("Failed to load model 5:", e)

model11_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/best_model_11.sav'

with open(model11_path, 'rb') as file:
    print(file.read(100))
try:
    model11 = joblib.load(model11_path)
    print("model 11 loaded successfully!")
except Exception as e:
    print("Failed to load model 11:", e)

model15_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/best_model_15.sav'

with open(model15_path, 'rb') as file:
    print(file.read(100))
try:
    model15 = joblib.load(model15_path)
    print("model 15 loaded successfully!")
except Exception as e:
    print("Failed to load model 15:", e)

model18_path = '/Volumes/Maestria/GitHub/NEO/NEO/Exports/API/best_model_18.sav'

with open(model18_path, 'rb') as file:
    print(file.read(100))
try:
    model18 = joblib.load(model18_path)
    print("model 18 loaded successfully!")
except Exception as e:
    print("Failed to load model 18:", e)


#print(pipeline)

st.set_page_config(page_title='Near Earth Objects hazard prediction', layout='wide')

col10, col11, col12 = st.columns(3)

with col10:
    st.markdown('Carlos Patricio Castañeda Bustamante')
with col11:
    st.markdown('Maestria en ingenieria en computacion')
with col12:
    st.markdown('Universidad Autonoma de Chihuahua')

st.title('Near Earth Objects hazard prediction 🌠')
st.header('Input Features')


option = st.selectbox(
    'How many features do you want to use?',
    ('5', '11', '15', '18'))

st.write('You selected:', option)

if option == '5':

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input('Enter a name')
        absolute_magnitude = st.number_input('Enter a absolute magnitude of its intrinsic luminosity(9 to 34)', min_value= 9.0, max_value=34.0, value=10.0, format= '%.3f')
        estimated_diameter_min = st.number_input('Enter a minimum estimated diameter in kilometers (0.001 to 40)', min_value=0.001, max_value=40.0, value=5.0, format= '%.3f')

    with col2:
        estimated_diameter_max = st.number_input('Enter a maximum estimated diameter in kilometers (0.001 to 90)', min_value=0.001, max_value=90.0, value=5.0, format= '%.3f')
        relative_velocity = st.number_input('Enter its velocity relative to Earth in km/h (200 to 300,000)', min_value=200.0, max_value=300000.0, value=500.0, format= '%.3f')
        miss_distance = st.number_input('Enter a miss distance relative to Earth in kilometers (6000 to 75,000,000)', min_value=6000.0, max_value=75000000.0, value=10000.0, format= '%.3f')

    col5, col6, col7 = st.columns([1,6,1])

    if st.button('Predict'):
        with col6:
            input_data = pd.DataFrame(
            
                {'absolute_magnitude': [absolute_magnitude], 'estimated_diameter_min': [estimated_diameter_min],
                'estimated_diameter_max': [estimated_diameter_max], 'relative_velocity': [relative_velocity], 'miss_distance': [miss_distance]},
                index=[0]
            )

            st.write(input_data)

            pipelined_data = pipeline5.transform(input_data)

            predictionlog  = model5.predict(pipelined_data)
            
            if predictionlog[0] == 1:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 89% accuracy is HIGH'
            else:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 89% accuracy is LOW'

            col3, col4, col8 = st.columns(3)

            with col4:
                st.markdown('<p style="text-align: center;"> '+ str(logprediction_text) , unsafe_allow_html=True)

elif option == '11':

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input('Enter a name')
        absolute_magnitude_h = st.number_input('Enter a absolute magnitude of its intrinsic luminosity(9 to 34)', min_value= 9.0, max_value=34.0, value=10.0, format= '%.3f')
        estimated_diameter_min = st.number_input('Enter a minimum estimated diameter in kilometers (0.001 to 40)', min_value=0.001, max_value=40.0, value=5.0, format= '%.3f')
        minimum_orbit_intersection = st.number_input('Enter a minimum orbit intersection in kilometers (1.52262e-07 to 0.568197)', min_value=0.0000000001, max_value=1.0, value=0.1, format= '%.3f')
        inclination = st.number_input('Enter its inclination in degrees (0 to 180)', min_value=0.0, max_value=180.0, value=0.0, format= '%.3f')
        aphelion_distance = st.number_input('Enter its aphelion distance in kilometers (0.7 to 680)', min_value=0.7, max_value=680.0, value=5.0, format= '%.3f')

    with col2:
        estimated_diameter_max = st.number_input('Enter a maximum estimated diameter in kilometers (0.001 to 90)', min_value=0.001, max_value=90.0, value=5.0, format= '%.3f')
        relative_velocity = st.number_input('Enter its velocity relative to Earth in km/h (200 to 300,000)', min_value=200.0, max_value=300000.0, value=500.0, format= '%.3f')
        miss_distance = st.number_input('Enter a miss distance relative to Earth in kilometers (6000 to 75,000,000)', min_value=6000.0, max_value=75000000.0, value=10000.0, format= '%.3f')
        eccentricity = st.number_input('Enter its eccentricity (0.003 to 0.9999)', min_value=0.003, max_value=0.9999, value=0.1, format= '%.3f')
        perihilion_distance = st.number_input('Enter a perihilion distance in kilometers (0.05 to 1.3)', min_value=0.05, max_value=1.3, value=0.1, format= '%.3f')
        estimated_diameter_average = st.number_input('Enter an estimated diameter average in kilometers (0 to 70)', min_value=0.0000001, max_value=70.0, value=5.0, format= '%.3f')

    col5, col6, col7 = st.columns([1,6,1])

    if st.button('Predict'):
        with col6:
            input_data = pd.DataFrame(
            
                {'absolute_magnitude_h': [absolute_magnitude_h], 'estimated_diameter_min': [estimated_diameter_min],
                'estimated_diameter_max': [estimated_diameter_max], 'relative_velocity.kilometers_per_hour': [relative_velocity], 'miss_distance.kilometers': [miss_distance], 'minimum_orbit_intersection': [minimum_orbit_intersection], 'inclination': [inclination], 'aphelion_distance': [aphelion_distance], 'eccentricity': [eccentricity], 'perihilion_distance': [perihilion_distance], 'estimated_diameter_average': [estimated_diameter_average]},
                index=[0]
            )

            st.write(input_data)

            pipelined_data = pipeline11.transform(input_data)

            predictionlog  = model11.predict(pipelined_data)
            
            if predictionlog[0] == 1:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.97% accuracy is HIGH'
            else:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.97% accuracy is LOW'

            col3, col4, col8 = st.columns(3)

            with col4:
                st.markdown('<p style="text-align: center;"> '+ str(logprediction_text) , unsafe_allow_html=True)

elif option == '15':


    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input('Enter a name')
        absolute_magnitude_h = st.number_input('Enter a absolute magnitude of its intrinsic luminosity(9 to 34)', min_value= 9.0, max_value=34.0, value=10.0, format= '%.3f')
        minimum_orbit_intersection = st.number_input('Enter a minimum orbit intersection in kilometers (1.52262e-07 to 0.568197)', min_value=0.0000000001, max_value=1.0, value=0.1, format= '%.3f')
        inclination = st.number_input('Enter its inclination in degrees (0 to 180)', min_value=0.0, max_value=180.0, value=0.0, format= '%.3f')
        aphelion_distance = st.number_input('Enter its aphelion distance in kilometers (0.7 to 680)', min_value=0.7, max_value=680.0, value=5.0, format= '%.3f')
        orbit_uncertainty = st.number_input('Enter its orbit uncertainty (0 to 10)', min_value=0.0, max_value=10.0, value=0.0, format= '%.3f')
        semi_major_axis = st.number_input('Enter its semi-major axis (0.5 to 360)', min_value=0.5, max_value=360.0, value=3.4, format= '%.3f')
        perihelion_argument = st.number_input('Enter its perihelion argument in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')
        mean_anomaly = st.number_input('Enter its mean anomaly in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')

    with col2:
        relative_velocity = st.number_input('Enter its velocity relative to Earth in km/h (200 to 300,000)', min_value=200.0, max_value=300000.0, value=500.0, format= '%.3f')
        miss_distance = st.number_input('Enter a miss distance relative to Earth in kilometers (6000 to 75,000,000)', min_value=6000.0, max_value=75000000.0, value=10000.0, format= '%.3f')
        eccentricity = st.number_input('Enter its eccentricity (0.003 to 0.9999)', min_value=0.003, max_value=0.9999, value=0.1, format= '%.3f')
        perihelion_distance = st.number_input('Enter a perihilion distance in kilometers (0.05 to 1.3)', min_value=0.05, max_value=1.3, value=0.1, format= '%.3f')
        jupiter_tisserand_invariant = st.number_input('Enter its jupiter tisserand invariant (-1 to 10)', min_value= -1.0, max_value=10.0, value=0.0, format= '%.3f')
        ascending_node_longitude = st.number_input('Enter its ascending node longitude in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')
        perihelion_time = st.number_input('Enter a perihelion time in days (2400000 to 2500000)', min_value=2400000.0, max_value=2500000.0, value=2400000.0, format= '%.3f')

    col5, col6, col7 = st.columns([1,6,1])

    if st.button('Predict'):
        with col6:
            input_data = pd.DataFrame(
            
                {'absolute_magnitude_h': [absolute_magnitude_h], 'relative_velocity.kilometers_per_hour': [relative_velocity], 'miss_distance.kilometers': [miss_distance], 'minimum_orbit_intersection': [minimum_orbit_intersection], 
                 'inclination': [inclination], 'aphelion_distance': [aphelion_distance], 'eccentricity': [eccentricity], 'perihelion_distance': [perihelion_distance], 'semi_major_axis': [semi_major_axis], 
                 'perihelion_argument': [perihelion_argument], 'mean_anomaly': [mean_anomaly], 'orbit_uncertainty': [orbit_uncertainty], 'perihelion_time': [perihelion_time], 'jupiter_tisserand_invariant': [jupiter_tisserand_invariant], 
                 'ascending_node_longitude': [ascending_node_longitude]},
                index=[0]
            )

            st.write(input_data)

            pipelined_data = pipeline15.transform(input_data)

            predictionlog  = model15.predict(pipelined_data)
            
            if predictionlog[0] == 1:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.96% accuracy is HIGH'
            else:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.96% accuracy is LOW'

            col3, col4, col8 = st.columns(3)

            with col4:
                st.markdown('<p style="text-align: center;"> '+ str(logprediction_text) , unsafe_allow_html=True)



elif option == '18':


    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input('Enter a name')
        absolute_magnitude_h = st.number_input('Enter a absolute magnitude of its intrinsic luminosity(9 to 34)', min_value= 9.0, max_value=34.0, value=10.0, format= '%.3f')
        estimated_diameter_min = st.number_input('Enter a minimum estimated diameter in kilometers (0.001 to 40)', min_value=0.001, max_value=40.0, value=5.0, format= '%.3f')
        minimum_orbit_intersection = st.number_input('Enter a minimum orbit intersection in kilometers (1.52262e-07 to 0.568197)', min_value=0.0000000001, max_value=1.0, value=0.1, format= '%.3f')
        inclination = st.number_input('Enter its inclination in degrees (0 to 180)', min_value=0.0, max_value=180.0, value=0.0, format= '%.3f')
        aphelion_distance = st.number_input('Enter its aphelion distance in kilometers (0.7 to 680)', min_value=0.7, max_value=680.0, value=5.0, format= '%.3f')
        orbit_uncertainty = st.number_input('Enter its orbit uncertainty (0 to 10)', min_value=0.0, max_value=10.0, value=0.0, format= '%.3f')
        semi_major_axis = st.number_input('Enter its semi-major axis (0.5 to 360)', min_value=0.5, max_value=360.0, value=3.4, format= '%.3f')
        perihelion_argument = st.number_input('Enter its perihelion argument in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')
        mean_anomaly = st.number_input('Enter its mean anomaly in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')

    with col2:
        estimated_diameter_max = st.number_input('Enter a maximum estimated diameter in kilometers (0.001 to 90)', min_value=0.001, max_value=90.0, value=5.0, format= '%.3f')
        relative_velocity = st.number_input('Enter its velocity relative to Earth in km/h (200 to 300,000)', min_value=200.0, max_value=300000.0, value=500.0, format= '%.3f')
        miss_distance = st.number_input('Enter a miss distance relative to Earth in kilometers (6000 to 75,000,000)', min_value=6000.0, max_value=75000000.0, value=10000.0, format= '%.3f')
        eccentricity = st.number_input('Enter its eccentricity (0.003 to 0.9999)', min_value=0.003, max_value=0.9999, value=0.1, format= '%.3f')
        perihelion_distance = st.number_input('Enter a perihilion distance in kilometers (0.05 to 1.3)', min_value=0.05, max_value=1.3, value=0.1, format= '%.3f')
        estimated_diameter_average = st.number_input('Enter an estimated diameter average in kilometers (0 to 70)', min_value=0.0000001, max_value=70.0, value=5.0, format= '%.3f')
        jupiter_tisserand_invariant = st.number_input('Enter its jupiter tisserand invariant (-1 to 10)', min_value= -1.0, max_value=10.0, value=0.0, format= '%.3f')
        ascending_node_longitude = st.number_input('Enter its ascending node longitude in degrees (0 to 360)', min_value=0.0, max_value=360.0, value=0.0, format= '%.3f')
        perihelion_time = st.number_input('Enter a perihelion time in days (2400000 to 2500000)', min_value=2400000.0, max_value=2500000.0, value=2400000.0, format= '%.3f')

    col5, col6, col7 = st.columns([1,6,1])

    if st.button('Predict'):
        with col6:
            input_data = pd.DataFrame(
            
                {'absolute_magnitude_h': [absolute_magnitude_h], 'estimated_diameter_min': [estimated_diameter_min],
                'estimated_diameter_max': [estimated_diameter_max], 'relative_velocity.kilometers_per_hour': [relative_velocity], 'miss_distance.kilometers': [miss_distance], 
                'minimum_orbit_intersection': [minimum_orbit_intersection], 'inclination': [inclination], 'aphelion_distance': [aphelion_distance], 'eccentricity': [eccentricity], 
                'perihelion_distance': [perihelion_distance], 'estimated_diameter_average': [estimated_diameter_average], 'mean_anomaly': [mean_anomaly], 'orbit_uncertainty': [orbit_uncertainty], 
                'perihelion_time': [perihelion_time], 'jupiter_tisserand_invariant': [jupiter_tisserand_invariant], 
                'ascending_node_longitude': [ascending_node_longitude], 'semi_major_axis': [semi_major_axis], 'perihelion_argument': [perihelion_argument]},
                index=[0]
            )

            st.write(input_data)

            pipelined_data = pipeline18.transform(input_data)

            predictionlog  = model18.predict(pipelined_data)
            
            if predictionlog[0] == 1:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.97% accuracy is HIGH'
            else:
                logprediction_text = 'The predicted hazard for ' + name + ' using an Adaboost model with an 99.97% accuracy is LOW'

            col3, col4, col8 = st.columns(3)

            with col4:
                st.markdown('<p style="text-align: center;"> '+ str(logprediction_text) , unsafe_allow_html=True)