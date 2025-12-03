# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:28:07 2025

@author: franc
"""
import pandas as pd
import numpy as np
from HybridReaders import read_wdbc, read_compass, read_german_credit, read_vehicle
import preprocess_and_training as pt
from joblib import load
from RuleTree import RuleTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 


basepath = "C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/"
save_path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'

df_name, df = read_wdbc(basepath = basepath)
X_tr, X_ts, y_tr, y_ts, feature_names, scaler = pt.split_data(df_name, df, save = True, save_path = save_path)
X_tr_df = pd.read_csv('Split_salvati/wdbc_X_tr.csv')

#TESTING split_data
print('HEAD: \n', X_tr_df.head())
print('COLUMNS: \n', X_tr_df.columns)
print('SCALER: ', scaler)

initial_cols = df.columns[:-1].tolist()
assert initial_cols == feature_names, 'Error: the feature names do not coincide'
assert X_tr_df.columns.tolist() == feature_names, 'Error: the csv columns do not coincide'
assert isinstance(X_tr, np.ndarray), 'Error: X_tr is not a np array'
assert X_tr_df.shape == X_tr.shape, 'Error: the X_tr and its csv dimensions do not coincide'


#TESTING train_models
df_name, df = read_vehicle(basepath = basepath)
save_path_models = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/vehicle'
X_tr, X_ts, y_tr, y_ts, feature_names = pt.split_data(df_name, df, save = False, save_path = save_path)

#pt.train_models(df_name, X_tr, X_ts, y_tr, y_ts, model = RuleTreeClassifier, save_path = save_path_models)
#pt.train_models(df_name, X_tr, X_ts, y_tr, y_ts, model = KNeighborsClassifier, save_path = save_path_models)
#pt.train_models(df_name, X_tr, X_ts, y_tr, y_ts, model = LogisticRegression, save_path = save_path_models)

model = load(f'{save_path_models}/vehicle_knn_3.joblib')
print(model)






























