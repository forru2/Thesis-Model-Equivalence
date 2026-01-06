# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 18:37:12 2025

@author: franc
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.base import clone
from performance import performance
import fairness as fn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score

def robustness(model, param_combo, X_tr, y_tr, measure = performance, measure_metric = accuracy_score, sensitive_feature_tr = None, n_folds = 5, **kwargs):
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds_indices = list(skf.split(X_tr, y_tr))

    tot_results = []
    for fold, (train_index, val_index) in enumerate(folds_indices):
        X_train, X_val = X_tr[train_index], X_tr[val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]
        
        mod = model.__class__(**param_combo)
        mod.fit(X_train, y_train)
        
        params = {'model': mod, 'X_ts': X_val, 'y_true': y_val}
        
        if measure is fn.fairness_explorer or measure is fn.advanced_fairness:
            sensitive_feature_val = sensitive_feature_tr[val_index]
            params['sensitive_features'] = sensitive_feature_val
            params['metric'] = measure_metric
       
        if measure is performance:
            params['measure'] = measure_metric
        
        
        result = measure(**params)
        tot_results.append(result)
        
    robustness = 1/(1 + np.std(tot_results))
    return robustness