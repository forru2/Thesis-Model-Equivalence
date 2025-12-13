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
from sklearn.model_selection import StratifiedKFold
import itertools
import fairness as fn
import performance as pf
import robustness as rb



#split df and save
def split_data(df_name, df, target_col = 'y', save = True, scale = True, test_size = 0.3, random_state = 42, save_path = './Split_salvati', **kwargs):
    
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



#otteniamo le possibili combo di parametri dato un dizionario di possibili valori per ogni parametro
def get_parameter_combos(param_grid:dict):
    
    keys, values = zip(*param_grid.items())
    parameter_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Tot_combos: {len(parameter_combos)}")
    return parameter_combos


def cross_val_metrics(model, model_name:str, X_tr, y_tr, X_ts, y_ts, parameter_combos, metrics_to_compute, 
                      file_path:str, 
                      fairness_sens_feat_tr, 
                      fairness_sens_feat_ts,
                      n_folds=5):
    
    #mi creo i path se non esistono
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    #if os.path.exists(file_path):
       # os.remove(file_path)
    
    #creo la lista di indici per le fold seguendo stratkfold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds_indices = list(skf.split(X_tr, y_tr))
    
    model_params_keys = model().get_params().keys()

    #per ogni combo di parametri (modello)
    for i, combo in enumerate(parameter_combos):
        
        print(f'CV model {i+1}/{len(parameter_combos)}')
        
        sum_results_val = {metric['name']: 0 for metric in metrics_to_compute}
        sum_results_test = {metric['name']: 0 for metric in metrics_to_compute}
        
        #per ogni fold divido prendendo la lista di indici già creata
        for fold, (train_index, val_index) in enumerate(folds_indices):
            
            X_train, X_val = X_tr[train_index], X_tr[val_index]
            y_train, y_val = y_tr[train_index], y_tr[val_index]
            
            #per calcolare la fairness mi serve una feature sensibile (colonna df) e devo regolarla al sample
            sens_feat_val = fairness_sens_feat_tr[val_index]
            
            #inizializzazione e fit del modello (il knn non aveva random state)
            if 'random_state' in model_params_keys:
                m = model(**combo, random_state=i)
            else:
                m = model(**combo)
            m.fit(X_train, y_train)
            
            #calcolo metriche per val e ts
            for metric in metrics_to_compute:
                params_val = {'model': m, 'X_ts': X_val, 'X_tr': X_train, 'y_tr': y_train, 'y_true': y_val, 'sensitive_features': sens_feat_val}
                params_val.update(metric.get('params', {}))
                
                params_test = {'model': m, 'X_ts': X_ts, 'X_tr': X_tr, 'y_tr': y_tr, 'y_true': y_ts, 'sensitive_features': fairness_sens_feat_ts}
                params_test.update(metric.get('params', {}))
                
                val_res = metric['func'](**params_val)
                test_res = metric['func'](**params_test)
                
                                        
                sum_results_val[metric['name']] += val_res
                sum_results_test[metric['name']] += test_res

        
        to_save = {'model_id': f'{model_name}_{i+1}'}
        
        for k in sum_results_val.keys():
            to_save[f'val_{k}'] = sum_results_val[k] / n_folds
            to_save[f'test_{k}'] = sum_results_test[k] / n_folds
            
        df_row = pd.DataFrame([to_save])
        #header = not os.path.exists(file_path) and i == 0
        header = (i == 0)
        df_row.to_csv(file_path, mode = 'a', header = header, index = False)
        

    print(f'Metrics saved in: \n{file_path}')
   


def full_train_metrics(model, model_name:str, df_name:str, X_tr, y_tr, X_ts, y_ts, parameter_combos, metrics_to_compute, metrics_path:str, model_dir:str, fairness_sens_feat, tree_root = None):
    
    #mi creo i path se non esistono e gestisco i csv già creati
    directory = os.path.dirname(metrics_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    #if os.path.exists(metrics_path):
        #os.remove(metrics_path)
        
    model_params_keys = model().get_params().keys()

    #per ogni combo di parametri (modello)
    for i, combo in enumerate(parameter_combos):
        
        print(f'Training model {i+1}/{len(parameter_combos)}')
        
        #inizializzazione e fit del modello (il knn non aveva random state)
        if 'random_state' in model_params_keys:
            m = model(**combo, random_state=i)
        else:
            m = model(**combo)
        m.fit(X_tr, y_tr)
        
        #salvo il modello
        model_path = os.path.join(model_dir, f'{df_name}_{model_name}_{i+1}.joblib')
        dump(m, model_path)
        
        #calcolo le metriche e salvo in csv
        to_save = {'model_id': f'{model_name}_{i+1}'}
        params = {'model': m, 'X_ts': X_ts, 'y_true': y_ts,
            'sensitive_features': fairness_sens_feat, 'node': tree_root}


        for metric in metrics_to_compute:
            params = {'model': m, 'X_ts': X_ts, 'X_tr': X_tr, 'y_tr': y_tr, 'y_true': y_ts, 'sensitive_features': fairness_sens_feat}
            params.update(metric.get('params', {}))
            
            result = metric['func'](**params)
            to_save[metric['name']] = result

        df_row = pd.DataFrame([to_save])
        df_row.to_csv(metrics_path, mode='a', header= (i == 0), index=False)

    print(f"Metrics saved in: {metrics_path}")
    print(f'Models saved in: \n{model_dir}')


if __name__ == '__main__':

    rtc_param_grid = {
        'criterion': ['gini', 'entropy'],               
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 
        'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 75, 100, 150], 
        'min_samples_leaf': [1, 2, 5, 10, 20]            
         }
    
    knn_param_grid = {
        'weights': ['uniform', 'distance'],               
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12], 
        'p': [round(n, 4) for n in np.linspace(1, 5, 50)], 
         }
    
    lr_param_grid = {
        'solver': ['saga'],                
        'penalty': ['l1', 'elasticnet'],   
        'C': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200],
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'max_iter': [100, 200, 400, 800, 1600],
        }
    
    metrics_to_compute = [
    {'name': 'accuracy', 'func': pf.performance, 'params': {'measure': pf.accuracy_score, 'average': 'weighted'}},
    {'name': 'f1', 'func': pf.performance, 'params': {'measure': pf.f1_score, 'average': 'weighted'}},
    {'name': 'precision', 'func': pf.performance, 'params': {'measure': pf.precision_score, 'average': 'weighted'}},
    {'name': 'recall', 'func': pf.performance, 'params': {'measure': pf.recall_score, 'average': 'weighted'}},
    {'name': 'equalized_odds', 'func': fn.advanced_fairness, 'params': {'metric': 'eo'}},
    {'name': 'cuae', 'func': fn.advanced_fairness, 'params': {'metric': 'cuae'}},
    {'name': 'dem_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.selection_rate}},
    {'name': 'equal_opportunity', 'func': fn.fairness_explorer, 'params': {'metric': fn.true_positive_rate}},
    {'name': 'predictive_equality', 'func': fn.fairness_explorer, 'params': {'metric': fn.false_positive_rate}},
    {'name': 'predictive_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.precision_score}},
    {'name': 'neg_predictive_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.negative_predictive_value}},
    {'name': 'acc_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.accuracy_score}}
    ]
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    X_ts_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')
    X_tr_df = pd.read_csv(f'{path}/german_credit_X_tr.csv')

    parameter_combos = get_parameter_combos(rtc_param_grid)
    cross_val_metrics(RuleTreeClassifier, 'rtc', X_tr, y_tr, X_ts, y_ts, parameter_combos, 
                      metrics_to_compute, 
                      file_path = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\rtc_mean_results.csv', 
                      fairness_sens_feat_tr = X_tr_df['personal_status'], 
                      fairness_sens_feat_ts = X_ts_df['personal_status']
                      )
    
    full_train_metrics(RuleTreeClassifier,
                       'rtc', 'german', X_tr, y_tr, X_ts, y_ts, parameter_combos, metrics_to_compute, 
                       metrics_path = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\final_results.csv', 
                       model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german', 
                       fairness_sens_feat =  X_ts_df['personal_status'])




