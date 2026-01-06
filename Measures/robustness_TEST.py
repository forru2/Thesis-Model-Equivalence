# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 21:06:44 2025

@author: franc
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.base import clone
from performance import performance
import fairness as fn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score
import robustness as rb
from joblib import load
import pandas as pd



def test_robustness(model, param_combo, X_tr, y_tr, measure = performance, measure_metric = accuracy_score, sensitive_feature_tr = None, n_folds = 5, **kwargs):
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds_indices = list(skf.split(X_tr, y_tr))

    tot_results = []
    for fold, (train_index, val_index) in enumerate(folds_indices):
        X_train, X_val = X_tr[train_index], X_tr[val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]
        
        mod = model.__class__(**param_combo)
        assert mod is not model
        mod.fit(X_train, y_train)
        
        params = {'model': mod, 'X_ts': X_val, 'y_true': y_val}
        
        if measure is fn.fairness_explorer or measure is fn.advanced_fairness:
            sensitive_feature_val = sensitive_feature_tr[val_index]
            params['sensitive_features'] = sensitive_feature_val
            params['metric'] = measure_metric
            assert list(params.keys()) == ['model', 'X_ts', 'y_true', 'sensitive_features', 'metric']
            
       
        if measure is performance:
            params['measure'] = measure_metric
            assert list(params.keys()) == ['model', 'X_ts', 'y_true', 'measure']
        
        
        result = measure(**params)
        tot_results.append(result)
     
    assert len(tot_results) == n_folds
    robustness = 1/(1 + np.std(tot_results))
    print('robustness: ', robustness)


if __name__ == '__main__':
    
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    model = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\000d875ef474857b97f33e8763e4bbb9.joblib')
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    X_tr_df = pd.read_csv(f'{path}/german_credit_X_tr.csv')
    sensitive_feature_tr = X_tr_df['personal_status']
    
    param_combo = model.get_params()
    
    test_robustness(model, param_combo, X_tr, y_tr, measure = performance, metric = accuracy_score)
    test_robustness(model, param_combo, X_tr, y_tr, measure = fn.fairness_explorer, metric = fn.selection_rate, sensitive_feature_tr = sensitive_feature_tr)
    
    
    
    
    
    
    
    