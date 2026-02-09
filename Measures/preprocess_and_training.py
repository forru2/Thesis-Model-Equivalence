# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:44:00 2025

@author: franc
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
#import random as rd
#from joblib import dump
#import sys
import os
#from sklearn.model_selection import StratifiedKFold
import itertools
import fairness as fn
import performance as pf
import robustness as rb
import json
import hashlib
#import os
import joblib
from Utils import convert_standard_python





#split df and save
def split_data(df_name, df, target_col = 'y', save = True, scale = True, test_size = 0.3, val_size = 0.2, random_state = 42, save_path = '.', **kwargs):
    
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size = val_size, random_state = 0, stratify = y_tr)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_ts = scaler.transform(X_ts)
    
    to_save_X = {
        f'{df_name}_X_tr': X_tr, 
        f'{df_name}_X_ts': X_ts,
        f'{df_name}_X_train': X_train,
        f'{df_name}_X_val': X_val
        }
    to_save_y = {
        f'{df_name}_y_tr': y_tr, 
        f'{df_name}_y_ts': y_ts,
        f'{df_name}_y_train': y_train,
        f'{df_name}_y_val': y_val
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
    
    return X_train, X_ts, X_val, y_train, y_ts, y_val, feature_names



#otteniamo le possibili combo di parametri dato un dizionario di possibili valori per ogni parametro
def get_parameter_combos(param_grid:dict):
    
    keys, values = zip(*param_grid.items())
    parameter_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Tot_combos: {len(parameter_combos)}")
    return parameter_combos



def get_file_name(d:dict):    
    d = convert_standard_python(d)
    initial_string = json.dumps(d, sort_keys = True).encode('utf-8')
    encoded_name = hashlib.md5(initial_string).hexdigest()
    return encoded_name


def get_metrics(model, X_tr, y_tr, X_ts, y_ts, metrics_to_compute, 
                                   sens_feat_tr, sens_feat_ts, param_combo, prefix=''):    
    results = {}     
    for metric in metrics_to_compute:
        params = {
            'model': model, 'X_ts': X_ts, 'X_tr': X_tr, 'y_tr': y_tr, 'y_true': y_ts, 
            'sensitive_features': sens_feat_ts, 'sensitive_feature_tr': sens_feat_tr,
            'param_combo': param_combo
        }
        params.update(metric.get('params', {}))
        results[f"{prefix}{metric['name']}"] = metric['func'](**params)
        
    return results
    



def model_results(model, X_train, X_ts, X_val, y_train, y_ts, y_val, sens_feat_train, 
                  sens_feat_ts, sens_feat_val, parameter_combo: dict,
                  metrics_to_compute,
                  random_state = None, noise_function = None):
    
    
    m_params = model().get_params()
    m = model(**parameter_combo, random_state = random_state) if 'random_state' in m_params else model(**parameter_combo) 
    m.fit(X_train, y_train)
    
    if noise_function:
        m = noise_function(m, random_state)
        
    res_val = get_metrics(model = m, X_tr = X_train, y_tr = y_train, X_ts = X_val, y_ts = y_val, 
                          metrics_to_compute = metrics_to_compute, 
                          sens_feat_tr = sens_feat_train, sens_feat_ts = sens_feat_val, 
                          param_combo = parameter_combo, prefix="val_")
    
    res_test = get_metrics(model = m, X_tr = X_train, y_tr = y_train, X_ts = X_ts, y_ts = y_ts,
                           metrics_to_compute = metrics_to_compute, 
                           sens_feat_tr = sens_feat_train, sens_feat_ts = sens_feat_ts, 
                           param_combo = parameter_combo, prefix="test_")
    
    return m, res_val, res_test    



def results_manager(model, X_train, X_ts, X_val, y_train, y_ts, y_val, parameter_combo:dict, metrics_to_compute, 
                    sens_feat_train, sens_feat_ts, sens_feat_val,
                    output_dir='.', random_state = None, model_dir = '.',
                    param_combo_processed = None, add_rs = False, noise_function = None):
    
    model_params = param_combo_processed if param_combo_processed is not None else parameter_combo
    d = {
        'model_type': model.__name__,
        'params': parameter_combo,
        'rs': random_state
        }
    file_name = get_file_name(d)
    file_path = os.path.join(output_dir, f'{file_name}.csv')
    if os.path.exists(file_path): 
        return pd.read_csv(file_path)
    
       
    m, res_val, res_test = model_results(
                            model = model, X_train = X_train, X_ts = X_ts, X_val = X_val, y_train = y_train, y_ts = y_ts, y_val = y_val,
                            sens_feat_train = sens_feat_train, sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                            random_state = random_state, parameter_combo = model_params, 
                            metrics_to_compute = metrics_to_compute, noise_function = noise_function
                            ) 
    
    res_dict = {**res_val, **res_test}
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(m, os.path.join(model_dir, f'{file_name}.joblib'))

    results = {'model_id': file_name, 'model_type': model.__name__, **parameter_combo, **res_dict}
    if add_rs:
        results['random_state_seed'] = random_state
        
    os.makedirs(output_dir, exist_ok = True) 
    df = pd.DataFrame([results])
    df.to_csv(file_path, index = False)
    return df
    



#mi fa la grid search di un modello e mi rende i parametri ottimali con le metriche di assessment
def grid(X_tr, y_tr, param_grid:dict, model, cv_folds = 5, n_jobs = -1):
    
    
    f1_0 = make_scorer(f1_score, pos_label = 0, zero_division = 0)
    f1_1 = make_scorer(f1_score, pos_label = 1, zero_division = 0)
    
    gs = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv_folds, n_jobs = n_jobs,
                      scoring = {'accuracy': 'accuracy', 'f1': 'f1_macro', 'f1_0': f1_0, 'f1_1': f1_1},
                      refit = 'f1',
                      return_train_score = True,
                      verbose =  0)
    gs.fit(X_tr, y_tr)
    idx = gs.best_index_
    results = {
        'best_params': gs.best_params_,
        'train_accuracy': gs.cv_results_['mean_train_accuracy'][idx],
        'val_accuracy': gs.cv_results_['mean_test_accuracy'][idx],
        'train_f1': gs.cv_results_['mean_train_f1'][idx],
        'val_f1': gs.cv_results_['mean_test_f1'][idx],
        'val_f1_class_0': gs.cv_results_['mean_test_f1_0'][idx],
        'val_f1_class_1': gs.cv_results_['mean_test_f1_1'][idx]       
    }
    return results



if __name__ == '__main__':

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
    {'name': 'acc_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.accuracy_score}},
    {'name': 'f1_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.f1_score}},
    {'name': 'precision_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.precision_score}},
    {'name': 'recall_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.recall_score}},
    {'name': 'dem_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.selection_rate}},
    {'name': 'equal_opp_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.true_positive_rate}},
    {'name': 'predictive_eq_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.false_positive_rate}},
    {'name': 'pred_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.precision_score}},
    {'name': 'neg_pred_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.negative_predictive_value}},
    {'name': 'eq_odds_robustness', 'func': rb.robustness, 'params': {'measure': fn.advanced_fairness, 'measure_metric': 'eo'}},
    {'name': 'eq_odds_robustness', 'func': rb.robustness, 'params': {'measure': fn.advanced_fairness, 'measure_metric': 'cuae'}}
    ]
    
    
    

    




    




    
    
    
    
    
    
    
