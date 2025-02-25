import pandas as pd
import json
import ast
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump, load
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.svm import SVC
import pickle as pkl
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.metrics import roc_curve, auc
import warnings
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from ngboost import NGBClassifier
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier




class append:
    def __init__(self,df2):

        self.df = df2 

    def name(self, data):

        t1 = ''
        for i in range(0,8):
            t1 += data[i]
        parts2 = t1.str.split(':')
        t2 = parts2.str[6]
        parts3 = t2.str.split('\'')
        name = parts3.str[1]
        return name
    

    def designation(self, data):

        specific = 'name_limited'
        contains_string = self.parts1.apply(lambda x: specific in x[4] if len(x) > 4 else False)

        if contains_string.any():
            name_limited = data[4]
            designation = data[5]
            nasa_jpl_url = data[6]
            abs_magnitude_h = data[7]
            diameter = ' '
            for i in range(8,16):
                diameter += data[i] + ','
            hazard = data[16]
        else:
            name_limited = ' '
            designation = data[4]
            nasa_jpl_url = data[5]
            abs_magnitude_h = data[6]
            diameter = ' '
            for i in range(7,15):
                diameter += data[i] + ','
            hazard = data[15]


        return name_limited, designation, nasa_jpl_url, abs_magnitude_h, diameter, hazard
    
    def orbit(self, data):

        orbit = ' '
        orbit = data[1]
        parts = orbit.str.split(',')
        orbit_data1 = ' '
        for i in range(1,26):
            orbit_data1 += parts.str[i].astype(str) + ','
        orbit_data1 = orbit_data1.str.rstrip(',')
        parts2 = orbit_data1.str.split(':')
        orbit_data = ' '
        for i in range(1,30):
            orbit_data += parts2.str[i].astype(str) + ':'
        orbit_data = orbit_data.str.rstrip(':')
        #orbit_data = orbit_data.apply(ast.literal_eval)

        is_sentry_object1 = ' '
        is_sentry_object1 = parts.str[26]
        parts1 = is_sentry_object1.str.split(':')
        is_sentry_object = parts1.str[1].astype(str)
        is_sentry_object = is_sentry_object.str.rstrip('}')
        is_sentry_object = is_sentry_object.str.lower() == 'true'
        is_sentry_object = is_sentry_object.astype(bool)

        return orbit_data, is_sentry_object

    def separation(self):
    

        # Assuming df2 is already defined
        df_result = pd.DataFrame()  # Initialize an empty DataFrame to store results

        for i in range(len(self.df)):
            df2t = self.df.iloc[[i]]  # Process each row
            df2t = pd.DataFrame(df2t)
            df2t['data'] = df2t['neo_data']
            df2t = df2t.drop(columns=['neo_data'])
            df2t = df2t.astype(str)

            parts = df2t['data'].str.split(' ')
            parts1 = df2t['data'].str.split(',')
            partsorbit = df2t['data'].str.split(']')
            self.parts1 = parts1
            name_limited, designation, nasa_jpl_url, abs_magnitude_h, diameter, hazard = self.designation(parts1.str)
            orbit_data, is_sentry_object = self.orbit(partsorbit.str)
            
            df2t['links'] = parts.str[1] + parts.str[2]
            df2t['id'] = parts.str[4]
            df2t['id'] = df2t['id'].str.replace('\'', '')
            df2t['id'] = df2t['id'].str.replace(',', '')
            df2t['id'] = df2t['id'].astype(int)


            df2t['neo_reference_id'] = parts.str[6]
            df2t['neo_reference_id'] = df2t['neo_reference_id'].str.replace('\'', '')
            df2t['neo_reference_id'] = df2t['neo_reference_id'].str.replace(',', '')
            df2t['neo_reference_id'] = df2t['neo_reference_id'].astype(int)


            df2t['name'] = self.name(parts1.str)


            df2t['name_limited'] = name_limited
            parts = df2t['name_limited'].str.split(' ')
            df2t['name_limited'] = parts.str[2]
            df2t['name_limited'] = df2t['name_limited'].astype(str)
            df2t['name_limited'] = df2t['name_limited'].str.replace('\'', '')


            df2t['designation'] = designation
            parts = df2t['designation'].str.split(' ')
            df2t['designation'] = parts.str[2]
            df2t['designation'] = df2t['designation'].astype(str)
            df2t['designation'] = df2t['designation'].str.replace('\'', '')


            df2t['nasa_jpl_url'] = nasa_jpl_url
            parts = df2t['nasa_jpl_url'].str.split(' ')
            df2t['nasa_jpl_url'] = parts.str[2]
            df2t['nasa_jpl_url'] = df2t['nasa_jpl_url'].astype(str)
            df2t['nasa_jpl_url'] = df2t['nasa_jpl_url'].str.replace('\'', '')


            df2t['absolute_magnitude_h'] = abs_magnitude_h
            parts = df2t['absolute_magnitude_h'].str.split(' ')
            df2t['absolute_magnitude_h'] = parts.str[2]
            df2t['absolute_magnitude_h'] = df2t['absolute_magnitude_h'].astype(str)
            df2t['absolute_magnitude_h'] = df2t['absolute_magnitude_h'].str.replace('\'', '')
            df2t['absolute_magnitude_h'] = df2t['absolute_magnitude_h'].astype(float)


            df2t['estiated_diameter1'] = diameter
            df2t['estiated_diameter1'] = df2t['estiated_diameter1'].str.rstrip(',')
            parts = df2t['estiated_diameter1'].str.split(' ')

            df2t['estimated_diameter'] = ' '
            for i in range(3,23):
                df2t['estimated_diameter'] += parts.str[i] + ' '
            df2t.drop(columns=['estiated_diameter1'], inplace=True)


            df2t['is_potentially_hazardous_asteroid'] = hazard
            parts = df2t['is_potentially_hazardous_asteroid'].str.split(' ')
            df2t['is_potentially_hazardous_asteroid'] = parts.str[2]
            df2t['is_potentially_hazardous_asteroid'] = df2t['is_potentially_hazardous_asteroid'].astype(str)
            df2t['is_potentially_hazardous_asteroid'] = df2t['is_potentially_hazardous_asteroid'].str.lower() == 'true'
            df2t['is_potentially_hazardous_asteroid'] = df2t['is_potentially_hazardous_asteroid'].astype(bool)

            parts1 = df2t['data'].str.split('[')
            df2t['approachdata'] = ' '
            df2t['approachdata'] = parts1.str[1]
            parts2 = df2t['approachdata'].str.split(']')
            df2t['close_approach_data'] = ' '
            df2t['close_approach_data'] = parts2.str[0]
            df2t['close_approach_data'] = df2t['close_approach_data'].apply(lambda x: f'[{x}]')
            df2t['close_approach_data'] = df2t['close_approach_data'].apply(ast.literal_eval)
            # df2t['close_approach_data'] = df2t['close_approach_data'].apply(lambda x: ' ' if x == [] else x)
            df2t = df2t[df2t['close_approach_data'].apply(lambda x: x != [])]

            df2t.drop(columns=['approachdata'], inplace=True)


            # df2t['close_approach_data'] = df2t['close_approach_data'].apply(ast.literal_eval)



            df2t['orbital_data'] = orbit_data


            df2t['is_sentry_object'] = is_sentry_object
            
                        


            df_result = pd.concat([df_result, df2t], ignore_index=True)  # Append to the result DataFrame

        df_result = df_result.drop(columns=['data'])
        df_result.dropna(subset=['close_approach_data'], inplace=True)
        self.df_result = df_result

        return self.df_result
    
    def concat(self):

        df_result = self.separation()

        df = pd.read_csv('/Volumes/Maestria/GitHub/NEO/NEO/API_test/neo_data.csv')

        df = pd.concat([df, df_result], ignore_index=True)

        return df
    
class OverallProcessor:

    def __init__(self, df):
        self.df = df

    def explode(self):

        def explode_approach(self):
            self.df['close_approach_data'] = self.df['close_approach_data'].apply(ast.literal_eval)
            self.df = self.df.explode("close_approach_data").reset_index(drop=True)
            normalized_close_approach_data = pd.json_normalize(self.df['close_approach_data'])
            self.df = pd.concat([self.df.drop(columns=['close_approach_data']), normalized_close_approach_data], axis=1)
            
        
        def clean_diameter(self):
                self.df.drop(columns=['neo_reference_id', 'name_limited', 'links', 'nasa_jpl_url'], inplace=True)
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('\'', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('{', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('}', '')
                parts = self.df['estimated_diameter'].str.split(',')
                self.df['estimated_diameter'] = parts.str[0] + parts.str[1]
                parts = self.df['estimated_diameter'].str.split(':')
                self.df['estimated_diameter'] = parts.str[1] + parts.str[2] + parts.str[3] 
                parts = self.df['estimated_diameter'].str.split(' ')
                self.df['estimated_diameter_min'] = parts.str[2]
                self.df['estimated_diameter_max'] = parts.str[4]
                self.df.drop(columns=['estimated_diameter'], inplace=True)


        def clean_orbits(self):
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('\'', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('{', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('}', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace(']', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('[', '')
            parts = self.df['orbital_data'].str.split(',')
            self.df['extracted_orbital_data'] = parts.str[7]+parts.str[10]+parts.str[12]+parts.str[15]+parts.str[17]
            parts = self.df['extracted_orbital_data'].str.split(' ')
            self.df['minimum_orbit_intersection'] = parts.str[2]
            self.df['eccentricity'] = parts.str[4]
            self.df['inclination'] = parts.str[6]
            self.df['perihilion_distance'] = parts.str[8]
            self.df['aphelion_distance'] = parts.str[10]
            self.df.drop(columns=['orbital_data', 'extracted_orbital_data'], inplace=True)    

        explode_approach(self)
        clean_diameter(self)
        clean_orbits(self)
        return self.df
    
    def clean(self):

        def explode_approach(self):
            self.df['close_approach_data'] = self.df['close_approach_data'].apply(ast.literal_eval)
            self.df = self.df.explode("close_approach_data").reset_index(drop=True)
            normalized_close_approach_data = pd.json_normalize(self.df['close_approach_data'])
            self.df = pd.concat([self.df.drop(columns=['close_approach_data']), normalized_close_approach_data], axis=1)
            
        
        def clean_diameter(self):
                self.df.drop(columns=['neo_reference_id', 'name_limited', 'links', 'nasa_jpl_url'], inplace=True)
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('\'', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('{', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('}', '')
                parts = self.df['estimated_diameter'].str.split(',')
                self.df['estimated_diameter'] = parts.str[0] + parts.str[1]
                parts = self.df['estimated_diameter'].str.split(':')
                self.df['estimated_diameter'] = parts.str[1] + parts.str[2] + parts.str[3] 
                parts = self.df['estimated_diameter'].str.split(' ')
                self.df['estimated_diameter_min'] = parts.str[2].astype(float)
                self.df['estimated_diameter_max'] = parts.str[4].astype(float)
                self.df.drop(columns=['estimated_diameter'], inplace=True)


        def clean_orbits(self):
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('\'', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('{', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('}', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace(']', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('[', '')
            parts = self.df['orbital_data'].str.split(',')
            self.df['extracted_orbital_data'] = parts.str[7]+parts.str[10]+parts.str[12]+parts.str[15]+parts.str[17]
            parts = self.df['extracted_orbital_data'].str.split(' ')
            self.df['minimum_orbit_intersection'] = parts.str[2].astype(float)
            self.df['eccentricity'] = parts.str[4].astype(float)
            self.df['inclination'] = parts.str[6].astype(float)
            self.df['perihilion_distance'] = parts.str[8].astype(float)
            self.df['aphelion_distance'] = parts.str[10].astype(float)
            self.df.drop(columns=['orbital_data', 'extracted_orbital_data'], inplace=True)   


        def clean_df(self):
            self.df.drop(columns=['id', 'name', 'designation', 'is_sentry_object', 'close_approach_date', 'close_approach_date_full', 'epoch_date_close_approach', 'orbiting_body', 'relative_velocity.kilometers_per_second', 'relative_velocity.miles_per_hour', 'miss_distance.astronomical', 'miss_distance.lunar', 'miss_distance.miles' ], inplace=True)
            self.df = self.df.rename(columns={'is_potentially_hazardous_asteroid': 'is_hazardous'}) 
            estimated_diameter_average = (self.df['estimated_diameter_min'].astype(float) + self.df['estimated_diameter_max'].astype(float)) / 2
            self.df['estimated_diameter_average'] = estimated_diameter_average

        # def encoder(self):
        #     le = LabelEncoder()
        #     self.df['is_hazardous'] = le.fit_transform(self.df['is_hazardous'])

        explode_approach(self)
        clean_diameter(self)
        clean_orbits(self)
        clean_df(self)
        # encoder(self)
        return self.df  
    
    def clean2(self):

        def explode_approach(self):
            self.df['close_approach_data'] = self.df['close_approach_data'].apply(ast.literal_eval)
            self.df = self.df.explode("close_approach_data").reset_index(drop=True)
            normalized_close_approach_data = pd.json_normalize(self.df['close_approach_data'])
            self.df = pd.concat([self.df.drop(columns=['close_approach_data']), normalized_close_approach_data], axis=1)
            
        
        def clean_diameter(self):
                self.df.drop(columns=['neo_reference_id', 'name_limited', 'links', 'nasa_jpl_url'], inplace=True)
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('\'', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('{', '')
                self.df['estimated_diameter'] = self.df['estimated_diameter'].str.replace('}', '')
                parts = self.df['estimated_diameter'].str.split(',')
                self.df['estimated_diameter'] = parts.str[0] + parts.str[1]
                parts = self.df['estimated_diameter'].str.split(':')
                self.df['estimated_diameter'] = parts.str[1] + parts.str[2] + parts.str[3] 
                parts = self.df['estimated_diameter'].str.split(' ')
                self.df['estimated_diameter_min'] = parts.str[2].astype(float)
                self.df['estimated_diameter_max'] = parts.str[4].astype(float)
                self.df.drop(columns=['estimated_diameter'], inplace=True)


        def clean_orbits(self):
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('\'', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('{', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('}', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace('[', '')
            self.df['orbital_data'] = self.df['orbital_data'].str.replace(']', '')
            parts = self.df['orbital_data'].str.split(',')
            self.df['extracted_orbital_data'] = parts.str[6:9].astype(str).str.cat(parts.str[10:20].astype(str), sep='')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace('\'', '')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace('{', '')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace('}', '')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace('[', '')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace(']', '')
            self.df['extracted_orbital_data'] = self.df['extracted_orbital_data'].str.replace(',', '')
            parts2 = self.df['extracted_orbital_data'].str.split(' ')
            self.df['orbit_uncertainty'] = parts2.str[2].astype(float)
            self.df['minimum_orbit_intersection'] = parts2.str[5].astype(float)
            self.df['jupiter_tisserand_invariant'] = parts2.str[8].astype(float)
            self.df['eccentricity'] = parts2.str[10].astype(float)
            self.df['semi_major_axis'] = parts2.str[13].astype(float)
            self.df['inclination'] = parts2.str[16].astype(float)
            self.df['ascending_node_longitude'] = parts2.str[19].astype(float)
            # df2t['orbital_period'] = parts2.str[22]
            self.df['perihelion_distance'] = parts2.str[25].astype(float)
            self.df['perihelion_argument'] = parts2.str[28].astype(float)
            self.df['aphelion_distance'] = parts2.str[31].astype(float)
            self.df['perihelion_time'] = parts2.str[34].astype(float)
            self.df['mean_anomaly'] = parts2.str[37].astype(float)
            self.df.drop(['orbital_data', 'extracted_orbital_data'], axis=1, inplace=True)


        def clean_df(self):
            self.df.drop(columns=['id', 'name', 'designation', 'is_sentry_object', 'close_approach_date', 'close_approach_date_full', 'epoch_date_close_approach', 'orbiting_body', 'relative_velocity.kilometers_per_second', 'relative_velocity.miles_per_hour', 'miss_distance.astronomical', 'miss_distance.lunar', 'miss_distance.miles' ], inplace=True)
            self.df = self.df.rename(columns={'is_potentially_hazardous_asteroid': 'is_hazardous'}) 
            estimated_diameter_average = (self.df['estimated_diameter_min'].astype(float) + self.df['estimated_diameter_max'].astype(float)) / 2
            self.df['estimated_diameter_average'] = estimated_diameter_average

        # def encoder(self):
        #     le = LabelEncoder()
        #     self.df['is_hazardous'] = le.fit_transform(self.df['is_hazardous'])

        explode_approach(self)
        clean_diameter(self)
        clean_orbits(self)
        clean_df(self)
        # encoder(self)
        return self.df  
    
    def clean3(self):
    
        self.df.drop(columns=['neo_id', 'name', 'orbiting_body'], inplace=True)

        self.df.dropna(inplace=True)
        # le = LabelEncoder()
        # self.df['is_hazardous'] = le.fit_transform(self.df['is_hazardous'])

        return self.df
    
class scalesplit:
    def __init__(self, df):
        self.df = df


    def ttsplitsmote(self):
                
        df_target = self.df['is_hazardous']
        df_features = self.df.drop(columns=['is_hazardous'])

        le = LabelEncoder()


        num_features = len(df_features.columns)

        if num_features == 5:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'])
                ])
        elif num_features == 11:    
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers','minimum_orbit_intersection', 'eccentricity', 'inclination', 'perihilion_distance', 'aphelion_distance', 'estimated_diameter_average'])
            ])

        elif num_features == 15:       
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly'])
            ])
        elif num_features == 18:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly', 'estimated_diameter_min', 'estimated_diameter_max', 'estimated_diameter_average'])
            ])

        pipeline = Pipeline([
            ('preprocess', preprocess)])


        X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)


        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        sm = SMOTE(sampling_strategy='minority', random_state=42)
        oversampled_X, oversampled_Y = sm.fit_resample(X_train, y_train)

        X_train = oversampled_X
        y_train = oversampled_Y



        return X_train, X_test, y_train, y_test
    
    def ttsplitadasyn(self):
                
        df_target = self.df['is_hazardous']
        df_features = self.df.drop(columns=['is_hazardous'])


        le = LabelEncoder()


        num_features = len(df_features.columns)

        if num_features == 5:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'])
                ])
        elif num_features == 11:    
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers','minimum_orbit_intersection', 'eccentricity', 'inclination', 'perihilion_distance', 'aphelion_distance', 'estimated_diameter_average'])
            ])

        elif num_features == 15:       
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly'])
            ])
        elif num_features == 18:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly', 'estimated_diameter_min', 'estimated_diameter_max', 'estimated_diameter_average'])
            ])

        pipeline = Pipeline([
            ('preprocess', preprocess)])

        X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)


        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        ada = ADASYN(sampling_strategy='minority', random_state=42)
        oversampled_X, oversampled_Y = ada.fit_resample(X_train, y_train)

        X_train = oversampled_X
        y_train = oversampled_Y



        return X_train, X_test, y_train, y_test
    

    def ttsplitrus(self):
                
        df_target = self.df['is_hazardous']
        df_features = self.df.drop(columns=['is_hazardous'])

        le = LabelEncoder()


        num_features = len(df_features.columns)

        if num_features == 5:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'])
                ])
        elif num_features == 11:    
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers','minimum_orbit_intersection', 'eccentricity', 'inclination', 'perihilion_distance', 'aphelion_distance', 'estimated_diameter_average'])
            ])

        elif num_features == 15:       
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly'])
            ])
        elif num_features == 18:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly', 'estimated_diameter_min', 'estimated_diameter_max', 'estimated_diameter_average'])
            ])

        pipeline = Pipeline([
            ('preprocess', preprocess)])


        X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)


        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        rus = RandomUnderSampler(random_state=42)
        oversampled_X, oversampled_Y = rus.fit_resample(X_train, y_train)

        X_train = oversampled_X
        y_train = oversampled_Y



        return X_train, X_test, y_train, y_test
    

    def ttsplitimbalance(self):
                
        df_target = self.df['is_hazardous']
        df_features = self.df.drop(columns=['is_hazardous'])

        le = LabelEncoder()


        num_features = len(df_features.columns)

        if num_features == 5:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'])
                ])
        elif num_features == 11:    
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers','minimum_orbit_intersection', 'eccentricity', 'inclination', 'perihilion_distance', 'aphelion_distance', 'estimated_diameter_average'])
            ])

        elif num_features == 15:       
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly'])
            ])
        elif num_features == 18:
            preprocess = ColumnTransformer([
                ('scaler', StandardScaler(), ['absolute_magnitude_h', 'relative_velocity.kilometers_per_hour', 'miss_distance.kilometers', 'orbit_uncertainty', 'minimum_orbit_intersection','jupiter_tisserand_invariant', 'eccentricity', 'semi_major_axis', 'inclination', 'ascending_node_longitude', 'perihelion_distance', 'perihelion_argument', 'aphelion_distance', 'perihelion_time', 'mean_anomaly', 'estimated_diameter_min', 'estimated_diameter_max', 'estimated_diameter_average'])
            ])

        pipeline = Pipeline([
            ('preprocess', preprocess)])

        X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)


        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)


        return X_train, X_test, y_train, y_test
    
class LogRegression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
#        self.best_estimator_ = LogisticRegression(C=0.001, fit_intercept=False, n_jobs=8, random_state=0,solver='newton-cholesky', warm_start=True)
        
    def fit(self):

            def gridsearch():
                logReg = LogisticRegression(random_state=42)

                param_grid = {'solver': ['liblinear', 'newton-cholesky'],
              'penalty':['none', 'l2', 'l1', 'elasticnet'],
              'C':[0.001, 0.01, 0.1, 1, 10, 100],
              'n_jobs': [8],
              'random_state': [0, 42, 32],
              'fit_intercept': [True, False],
              'warm_start': [True, False]
}


                grid_search = GridSearchCV(logReg, param_grid, cv=5, verbose=0, n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)


#                self.best_estimator_ = LogisticRegression(C=0.001, fit_intercept=False, n_jobs=8, random_state=0,solver='liblinear', warm_start=True) 


                print(f'Best parameters: {grid_search.best_params_}')
                print(f'Best Score: {grid_search.best_score_}')
                print(f'Best Estimator: {grid_search.best_estimator_} ')

                return grid_search.best_estimator_

            def Regression(self):

                warnings.filterwarnings("ignore", message=".*'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'.*")

                k = 10
                kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

                best_model = gridsearch()
                best_model.fit(self.X_train, self.y_train)

                prediction = best_model.predict(self.X_test)

                accuracy = accuracy_score(self.y_test, prediction)
                recall = recall_score(prediction, self.y_test)
                f1 = f1_score(prediction, self.y_test)
                auc = roc_auc_score(self.y_test, prediction)

                print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {auc}\n')

                print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

                r2 =  cross_val_score(best_model,self.X_train,self.y_train,cv=kf,scoring='r2')
                print(f'Cross validation score: {r2}\n')

                np.mean(r2)

                print(f'Mean cross validation score: {np.mean(r2)}\n')

                cm = confusion_matrix(self.y_test, prediction)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.show()

                cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

                sns.heatmap(cm2, annot=True, cmap='Blues')

                
            # gridsearch()
            Regression(self)
    def predict(self, dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        logreg = self.best_estimator_.fit(self.X_train, self.y_train)

        prediction = logreg.predict(self.dfpred)

        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class supportvm:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.bestsvm = SVC(C=10000, random_state=42)

    def fit(self):

        # bestsvm = SVC(C=10000, random_state=42)

        # svmgrid = SVC()

        # param_grid = { 'C' : [2000, 5000, 10000],
        #      'gamma' : ['scale', 'auto']
        # }

        sgdc = SGDClassifier(random_state=42)

        param_grid = { 'loss' : ['modified_huber', 'perceptron', 'log_loss','squared_error', 'hinge'],
                      'penalty': ['l2', 'l1', 'elasticnet'],
                      'alpha': [0.0001, 0.001, 0.01],
                      'learning_rate': ['constant', 'adaptive'],
                      'eta0': [0.01,0.1,1]
        }


        grid_search = GridSearchCV(sgdc, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)



        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')


        k = 10 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # svm = SVC(C=10000, random_state=42, kernel='rbf', gamma='scale')
        svm = grid_search.best_estimator_
        svm.fit(self.X_train, self.y_train)

        prediction = svm.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(svm,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self, dfp):
    

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        svm = self.bestsvm.fit(self.X_train, self.y_train)

        prediction = svm.predict(self.dfpred)

        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class supportvm2:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.bestsvm = SVC(C=2000, gamma='auto', random_state=42)

    def fit(self):

        # bestsvm = SVC(C=10000, random_state=42)


        k = 5 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # svm = SVC(C=10000, random_state=42, kernel='rbf', gamma='scale')
        svm = self.bestsvm
        svm.fit(self.X_train, self.y_train)

        prediction = svm.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(svm,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):
    
        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        svm = self.bestsvm.fit(self.X_train, self.y_train)

        prediction = svm.predict(self.dfpred)

        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class RandomForest:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #self.randomforest = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy',random_state=42)
        
    def fit(self):

        warnings.filterwarnings("ignore", message=".*class_weight presets 'balanced' or 'balanced_subsample' are not recommended for warm_start if the fitted data differs from the full dataset.*", module="sklearn.ensemble._forest")


        randomfc = RandomForestClassifier(n_estimators=100, random_state=42)

        param_grid = { 'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2'], 
                'bootstrap': [True, False],
        }

        grid_search = GridSearchCV(randomfc, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)



        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')

        k = 10 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        randomforest = grid_search.best_estimator_
        randomforest.fit(self.X_train, self.y_train)

        prediction = randomforest.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')


        r2 =  cross_val_score(randomforest,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        rf = self.randomforest.fit(self.X_train, self.y_train)

        prediction = rf.predict(self.dfpred)

        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class PCAFeatures:
    def __init__(self, X_train, y_train, df):

        self.X_train = X_train
        self.y_train = y_train
        self.df = df

    def PCAfeatures(self):

        ydf = pd.DataFrame(self.y_train, columns= ['is_hazardous'])
        fd1 = self.df.drop(columns=['is_hazardous'])
        fd2 = pd.DataFrame(self.X_train, columns=fd1.columns)


        pca = PCA(n_components= 0.9, random_state=42)
        df_reduced = pca.fit_transform(fd2)

        print("Número de componentes:", pca.n_components_)

        feature_names = fd2.columns
        components = pca.components_
        df_components = pd.DataFrame(components, columns=feature_names, index=[f'PC{i+1}' for i in range(components.shape[0])])
        # print(df_components)

        df_pca = pd.DataFrame(df_reduced, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
        df_pca['class'] = ydf
        # print(df_pca.head())

        return df_pca

class xgbClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.bestxgb = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.01, random_state=42, gamma=0.1, alpha=0.1, min_child_weight=10)

    def fit(self):

        xgbgrid = xgb.XGBClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting rounds
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'max_depth': [3, 5, 7],  # Maximum tree depth for base learners
            'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight(hessian) needed in a child
            'gamma': [0, 0.1, 0.2]  # Minimum loss reduction required to make a further partition on a leaf node
            # 'subsample': [0.8, 1.0],  # Subsample ratio of the training instances
            # 'colsample_bytree': [0.8, 1.0],  # Subsample ratio of columns when constructing each tree
            # 'reg_alpha': [0, 0.01, 0.1],  # L1 regularization term on weights
            # 'reg_lambda': [1, 1.5, 2.0]  # L2 regularization term on weights
        }

        grid_search = GridSearchCV(xgbgrid, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)



        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')

        k = 10 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


        bestxgb = grid_search.best_estimator_
        bestxgb.fit(self.X_train, self.y_train)

        prediction = bestxgb.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')


        r2 =  cross_val_score(bestxgb,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        xgb = self.bestxgb.fit(self.X_train, self.y_train)

        prediction = xgb.predict(self.dfpred)


        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class GradientBoost:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #self.bestgb = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,random_state=42,max_depth=3)

    def fit(self):

        gbgrid = GradientBoostingClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting stages
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'max_depth': [3, 5, 7],  # Maximum tree depth for base learners
            'max_features': ['sqrt', 'log2', None]  # The number of features to consider when looking for the best split
        }

        grid_search = GridSearchCV(gbgrid, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')

        k = 10 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        bestgb = grid_search.best_estimator_
        bestgb.fit(self.X_train, self.y_train)

        prediction = bestgb.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(bestgb,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        gb = self.bestgb.fit(self.X_train, self.y_train)

        prediction = gb.predict(self.dfpred)


        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class AdaBoost:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #self.bestada = AdaBoostClassifier( RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', random_state=42),random_state=42)

    def fit(self):

        abgrid = AdaBoostClassifier( RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', random_state=42),random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of boosting stages
            'learning_rate': [0.01, 0.1, 0.5, 1.0],  # Weight applied to each classifier at each boosting iteration
            'algorithm': ['SAMME', 'SAMME.R']    # The number of features to consider when looking for the best split
        }

        grid_search = GridSearchCV(abgrid, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')


        k = 10
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


        bestada = grid_search.best_estimator_
        bestada.fit(self.X_train, self.y_train)

        prediction = bestada.predict(self.X_test) 

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(bestada,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        ada = self.bestada.fit(self.X_train, self.y_train)

        prediction = ada.predict(self.dfpred)


        return prediction

        # print(f'The prediction is {prediction}')
        # print('\n')

class ngboost:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #self.bestngb = NGBClassifier(n_estimators=100, random_state=42)

    def fit(self):

        ngbgrid = NGBClassifier(random_state = 42)

        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting stages
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'minibatch_frac': [0.5, 0.7, 1.0],
            'max_depth': [3, 5, 7]

        }

        grid_search = GridSearchCV(ngbgrid, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Estimator: {grid_search.best_estimator_} ')

        k = 10
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


        bestngb = grid_search.best_estimator_
        bestngb.fit(self.X_train, self.y_train)

        prediction = bestngb.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(bestngb,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        ngb = self.bestngb.fit(self.X_train, self.y_train)

        prediction = ngb.predict(self.dfpred)


        return prediction
        # print(f'The prediction is {prediction}')
        # print('\n')

class ANN:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.bestmlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)

    def fit(self):

        k = 10 
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        self.bestmlp.fit(self.X_train, self.y_train)

        prediction = self.bestmlp.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, prediction)
        recall = recall_score(prediction, self.y_test)
        f1 = f1_score(prediction, self.y_test)
        roc = roc_auc_score(self.y_test, prediction)

        print (f'The accuracy score is {accuracy}\n The recall score is {recall}\n The f1 score is {f1}\n The ROC AUC score is {roc}\n')

        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}\n')

        r2 =  cross_val_score(self.bestmlp,self.X_train,self.y_train,cv=kf,scoring='r2')
        print(f'Cross validation score: {r2}\n')

        np.mean(r2)

        print(f'Mean cross validation score: {np.mean(r2)}\n')

        cm = confusion_matrix(self.y_test, prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        cm2 = cm / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm2, annot=True, cmap='Blues')

    def predict(self,dfp):

        self.dfpred = dfp

        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message="class_weight presets 'balanced' or 'balanced_subsample'.* ", module="sklearn.ensemble._forest")

        mlp = self.bestmlp.fit(self.X_train, self.y_train)

        prediction = mlp.predict(self.dfpred)


        return prediction

        # print(f'The prediction is {prediction}')
        # print('\n')