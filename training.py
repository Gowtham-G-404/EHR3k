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
    
    
    test_scores = []
    train_scores = []
    
    for i in range(1,15):
    
        knn = KNeighborsClassifier(i)
        knn.fit(X_train,y_train)
        
        train_scores.append(knn.score(X_train,y_train))
        test_scores.append(knn.score(X_test,y_test))
        
    max_test_score = max(test_scores)
    test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
    temp = list(map(lambda x: x+1, test_scores_ind))
    
    #Setup a knn classifier with k neighbors
    model = KNeighborsClassifier(temp[0])
    
    model.fit(X_train,y_train)
    #knn.score(X_test,y_test)
    
    joblib.dump(model, 'diabetes_knn.pkl')
    return 0

    
    