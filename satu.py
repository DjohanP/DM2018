import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imputer import Imputer
impute = Imputer()
dataset = pd.read_csv('Dataset/First/first.csv', na_values='?',usecols=['trestbps', 'chol'])

#dataset['age']=dataset['age'].astype('float64')
dataset['trestbps']=dataset['trestbps'].astype('float64')
dataset['xhol']=dataset['chol'].astype('float64')   
#dataset['thalach']=dataset['thalach'].astype('float64')
#dataset['oldpeak']=dataset['oldpeak'].astype('float64')

dataset_imputed = impute.knn(X=dataset, column='chol', k=3)

