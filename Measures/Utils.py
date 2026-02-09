# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 13:11:57 2025

@author: franc
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, confusion_matrix, multilabel_confusion_matrix
from scipy.stats import kendalltau, spearmanr, pearsonr
import glob
import os
import pandas as pd

#prende un modello o una lista di modelli e ne calcola le predizioni e le probabilità per ogni classe
def predictions(models, X_ts, limit_to_two = True, **kwargs):
        
    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]
    
    if limit_to_two and len(models) > 2:
        raise ValueError(f'The function handles at most 2 models, you are comparing {len(models)} models')
    
    preds = np.array([model.predict(X_ts) for model in models])
    probs = np.array([model.predict_proba(X_ts) for model in models])
    
    if len(models) == 1:
        preds = preds[0]
        probs = probs[0]
        
    return preds, probs

#applica una metrica (corr, errore) facendo la differenza tra quelle che rendono tuple e quelle
#che rendono un valore (per gestire la correlazione che rende anche il p value)
def apply_distance_metric(x, y, metric, **kwargs): 
    result = metric(x, y) 
    if isinstance(result, tuple): 
        val = result[0] 
    else: 
        val = result
    #per gestire la divisione per 0 in caso di correlazione   
    if np.isnan(val):
        std_x = np.std(x)
        std_y = np.std(y)
        
        if std_x == 0 and std_y == 0:
            return 1.0
        else:
            return 0.0
    return val

def prediction_concordance_filter(models, X_ts, to_filter, y_true = None, concordant = True, **kwargs):
    
    preds, _ = predictions(models, X_ts)
    preds1, preds2 = preds
    
    if y_true is None:
        
        condition = (preds1 == preds2) if concordant  else (preds1 != preds2) 
    else:
        condition = (preds1 == y_true) & (preds2 == y_true) if concordant else (preds1 != y_true) & (preds2 != y_true)
    
    if not isinstance(to_filter, (list, tuple)):
        return to_filter[condition]
    else:
        return [f[condition] for f in to_filter]

    
#fa la concatenazione di df in csv nella stessa cartella anche se sono nelle rispettive sottocartelle
def concatenate_df(path, file_names, axis = 0):
    files_in_path = glob.glob(os.path.join(path, f'**/{file_names}.csv'), recursive = True)
    
    to_concat = []
    for file in files_in_path:
        df = pd.read_csv(file)
        to_concat.append(df)
    return pd.concat(to_concat, axis = axis, ignore_index = True)


def convert_standard_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_standard_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_standard_python(el) for el in obj]
    return obj


















