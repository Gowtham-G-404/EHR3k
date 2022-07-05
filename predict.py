# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:55:54 2022

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:44:29 2022

@author: admin
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    df = pd.read_csv("diabetes.csv")
    
    df_copy = df.copy(deep = True)
    df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    
    df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)
    df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)
    df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
    df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
    df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X =  pd.DataFrame(sc_X.fit_transform(df_copy.drop(["Outcome"],axis = 1),),
            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    y = df_copy.Outcome
    
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
    
    
    save_model = joblib.load('diabetes_knn.pkl')
    
    y_pred = save_model.predict(X_test)
    X_test['y_actual'] = y_test
    X_test['y_pred'] = y_pred
    
    X_test.to_csv("predict_results.csv")
    final_results = {}
    final_results["Actual Diagonosis"] = X_test['y_actual'] 
    final_results["Predicted Diagonosis"] = X_test['y_pred']   
    
    return final_results 
    