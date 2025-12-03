# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:50:42 2025

@author: franc
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
from fairlearn.metrics import selection_rate, MetricFrame, true_positive_rate, false_positive_rate



def negative_predictive_value(y_true, y_pred):     
    tn, _, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() 
    if (tn + fn) == 0:
        return 0.0
    return tn / (tn + fn)


#crea una serie pandas con metriche e rispettivi valori
def fairness_explorer(model, X_ts, y_true, sensitive_features, metrics: dict):
   
    y_pred = model.predict(X_ts)
    classes = np.unique(y_true)
    
    if len(classes) > 2:
        fairness_per_class = []
        for cl in classes:
            y_true_binary = (y_true == cl).astype(int)
            y_pred_binary = (y_pred == cl).astype(int)
            explorer = MetricFrame(
                                   metrics = metrics, 
                                   y_true = y_true_binary, 
                                   y_pred = y_pred_binary,
                                   sensitive_features = sensitive_features
                                   )
            fairness_per_class.append(explorer.difference())
        return fairness_per_class
    
    else:
        explorer = MetricFrame(
                               metrics = metrics, 
                               y_true = y_true, 
                               y_pred = y_pred,
                               sensitive_features = sensitive_features
                               )
        return explorer.difference()


#mi calcola eo e cuae come fa fairlearn, anche se possono essere visti già da fairness explorer
def advanced_fairness(model, X_ts, y_true, sensitive_features):
    
    differences = fairness_explorer(model,
                                    X_ts,
                                    y_true,
                                    sensitive_features,
                                    metrics = {
                                        'equal opportunity(TPR)': true_positive_rate,
                                        'predictive equality(FPR)': false_positive_rate,
                                        'predictive parity(precision)': precision_score,
                                        'negative predictive parity(NPV)': negative_predictive_value
                                        })
    
    if isinstance(differences, list):
        eq_odds_list = []
        cuae_list = []
        for el in differences:
            
            eq_odds = max(el['equal opportunity(TPR)'], el['predictive equality(FPR)'])
            cuae = max(el['predictive parity(precision)'], el['negative predictive parity(NPV)'])
            eq_odds_list.append(eq_odds)
            cuae_list.append(cuae)
    
        fairness_dict = {'equalized_odds': eq_odds_list, 'cond_use_accuracy_equality': cuae_list}
    else:
        eq_odds = max(differences['equal opportunity(TPR)'], differences['predictive equality(FPR)'])
        cuae = max(differences['predictive parity(precision)'], differences['negative predictive parity(NPV)'])
        
        fairness_dict = {'equalized_odds': eq_odds, 'cond_use_accuracy_equality': cuae}
    return fairness_dict


fairness_metrics = {
    'demographic parity(selection_rate)': selection_rate,
    'equal opportunity(TPR)': true_positive_rate,
    'predictive equality(FPR)': false_positive_rate,
    'predictive parity(precision)': precision_score,
    'negative predictive parity(NPV)': negative_predictive_value}

