# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 18:29:51 2021

@author: Tejas Phirke
"""


import numpy as np
import pickle
import pandas as pd

import streamlit as st 

from PIL import Image
data = pd.read_csv('diabetes.csv')
# checking of Null value present or not 
data.isnull() 
# Since there are 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
data_copy = data.copy(deep=True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Null values as NaN must be replace with value by mean, median depending upon distribution
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)


### Independent and Dependent features
X=data_copy.iloc[:,:-1]
y=data_copy.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


def Hello():
    return "Hello"


def diabetes_pred(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age):
    
    
    DiabetesPedigreeFunction = np.asarray(DiabetesPedigreeFunction, dtype='float64')
    BMI = np.asarray(BMI, dtype='float64')
   
    prediction= classifier.predict(sc.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]]))
    print(prediction)
    return prediction



def main():
    st.title("Diabetes Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:2px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Pregnancies = st.text_input("Pregnancies","Type Here")
    Glucose = st.text_input("Glucose","Type Here")
    BloodPressure = st.text_input("BloodPressure","Type Here")
    SkinThickness = st.text_input("SkinThickness","Type Here")
    Insulin = st.text_input("Insulin","Type Here")
    BMI = st.text_input("BMI","Type Here")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction","Type Here")
    Age = st.text_input("Age","Type Here")
    result=""
    

    if st.button("Predict"):
        result=diabetes_pred(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                             BMI, DiabetesPedigreeFunction, Age)
    st.subheader('Your Report: ')
    st.success('The output is {}'.format(result))
    
    if (result==1):
        st.subheader('You have Diabetes')
    else:
        st.subheader('You do not have Diabetes')
 

if __name__=='__main__':
    main()
