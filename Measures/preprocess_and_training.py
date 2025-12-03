# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:44:00 2025

@author: franc
"""

import pandas as pd
import numpy as np
from RuleTree import RuleTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random as rd
from joblib import dump
import sys
import os



#split df and save
def split_data(df_name, df, target_col = 'y', save = True, scale = True, test_size = 0.3, random_state = 42, save_path = './Split_salvati'):
    
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_ts = scaler.transform(X_ts)
    
    to_save_X = {
        f'{df_name}_X_tr': X_tr, 
        f'{df_name}_X_ts': X_ts 
        }
    to_save_y = {
        f'{df_name}_y_tr': y_tr, 
        f'{df_name}_y_ts': y_ts
        }
    
    if save:
        try:
            for key, el in to_save_X.items():
                pd.DataFrame(el, columns = feature_names).to_csv(os.path.join(save_path, f'{key}.csv'), index=False)
                
            for key, el in to_save_y.items():
                pd.DataFrame(el, columns = [target_col]).to_csv(os.path.join(save_path, f'{key}.csv'), index=False)
            
            print(f'saved in: {save_path}')
            
        except FileNotFoundError:
            raise FileNotFoundError(f'The directory {save_path} does not exist or it is wrong')
    
    return X_tr, X_ts, y_tr, y_ts, feature_names, scaler


#model training and saving
def train_models(df_name, X_tr, X_ts, y_tr, y_ts, model = RuleTreeClassifier, max_models = 5, min_dpth = 3, max_dpth = 20, save_path = './', max_nb = 5, min_nb = 1):
    
    if not os.path.exists(save_path):
         print(f"Directory '{save_path}' does not exist")
         sys.exit(1)
    
    np.random.seed(0)
    
    n_neighbors = np.random.choice(np.arange(min_nb, max_nb +1), size = max_models, replace = False)
    C = np.random.choice(np.logspace(-4, 4), size = max_models, replace = False)
    for i in range(max_models):
        
        if model is RuleTreeClassifier:
            model_name = 'rtc'
            m = model(
                max_depth = rd.randint(min_dpth, max_dpth),
                criterion = rd.choice(('gini', 'entropy')),
                prune_useless_leaves=True,
                random_state = i
            )
            params = ['max_depth', 'criterion']
            
        elif model is KNeighborsClassifier:
            model_name = 'knn'
            m = model(
                n_neighbors = n_neighbors[i],
                weights = rd.choice(('uniform','distance'))
            )
            params = ['n_neighbors', 'weights']

        elif model is LogisticRegression:
            model_name = 'lr'
            m = model(
                C = C[i],
                max_iter = 500,
                random_state = i
            )
            params = ['C']
            
        else:
            print(f"{model} not recognized")
            return

        m.fit(X_tr, y_tr)
        try:
            dump(m, os.path.join(save_path, f'{df_name}_{model_name}_{i+1}.joblib'))
        
        except Exception as e:
            print(f'Saving error: {e}')
            sys.exit(1)
 
        #solo per dare una prima occhiata veloce
        y_pred = m.predict(X_ts)
        print(f'{df_name}_{model_name}_{i+1} \naccuracy: {accuracy_score(y_ts, y_pred)}') 
        for par in params:
            print(f'{par}: {m.get_params()[par]}')
            

















