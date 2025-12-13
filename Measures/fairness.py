# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:50:42 2025

@author: franc
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
from fairlearn.metrics import selection_rate, MetricFrame, true_positive_rate, false_positive_rate



def negative_predictive_value(y_true, y_pred, **kwargs):     
    tn, _, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() 
    if (tn + fn) == 0:
        return 0.0
    return tn / (tn + fn)


#calcola le metriche di fairness
def fairness_explorer(model, X_ts, y_true, sensitive_features, metric, take_max = True, **kwargs):
    
    if isinstance(metric, dict):
        raise ValueError('the function computes on metric at a time')
    
    y_pred = model.predict(X_ts)
    classes = np.unique(y_true)
    
    if len(classes) > 2:
        fairness_per_class = []
        for cl in classes:
            y_true_binary = (y_true == cl).astype(int)
            y_pred_binary = (y_pred == cl).astype(int)
            explorer = MetricFrame(
                                   metrics = metric, 
                                   y_true = y_true_binary, 
                                   y_pred = y_pred_binary,
                                   sensitive_features = sensitive_features
                                   )
            fairness_per_class.append(explorer.difference())
        
        if take_max:
            return max(abs(np.array(fairness_per_class)))
        else:
            return fairness_per_class
    
    else:
        explorer = MetricFrame(
                               metrics = metric, 
                               y_true = y_true, 
                               y_pred = y_pred,
                               sensitive_features = sensitive_features
                               )
        return explorer.difference()


#mi calcola eo e cuae come fa fairlearn
def advanced_fairness(model, X_ts, y_true, sensitive_features, metric:str = 'eo', **kwargs):
    if metric == 'eo':
        meas1 = fairness_explorer(model, X_ts, y_true, sensitive_features, metric = true_positive_rate)
        meas2 = fairness_explorer(model, X_ts, y_true, sensitive_features, metric = false_positive_rate)
        
    elif metric == 'cuae':
        meas1 = fairness_explorer(model, X_ts, y_true, sensitive_features, metric = lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0))
        meas2 = fairness_explorer(model, X_ts, y_true, sensitive_features, metric = negative_predictive_value)
        
    else:
        raise ValueError(f"'{metric}' not supported. Choose between 'eo' and 'cuae'.")
        
    if isinstance(meas1, list):
        final_measure = [max(meas1[i], meas2[i]) for i in range(len(meas1))]
    
    else:
        final_measure = max(meas1, meas2)
        
    return final_measure



