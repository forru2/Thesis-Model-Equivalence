# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:57:47 2025

@author: franc
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from Utils import prediction_concordance_filter, apply_distance_metric, spearman
from sklearn.metrics import mean_absolute_error, mean_squared_error
from shap import TreeExplainer, LinearExplainer, KernelExplainer
from joblib import Parallel, delayed
import os
import joblib
import time 

#calcola la similarità tra due array delle importanze
#la uso sia in shaps_similarity che in feature_importance_similarity_rtc
#se aggregate = False rende un'array (n.inst,) o (n.inst, n.classes) in caso multiclasse
#calcola mse, mae, spearman e spearman_error [(1-spear)/2]
def importance_array_similarity(arr1, arr2, metric, aggregate = False, **kwargs):
    if metric in ['mae', mean_absolute_error]:
        tot_measures = np.mean(np.abs(arr1 - arr2), axis = 1)
    elif metric in ['mse', mean_squared_error]:
        tot_measures = np.mean((arr1 - arr2)**2, axis = 1)
    elif metric in ['spearman', spearman, spearmanr]:
        tot_measures = spearman(arr1, arr2)
    elif metric == 'spearman_error':
        tot_measures = (1-spearman(arr1, arr2))/2
    else:
        raise ValueError(f'{metric} not supported yet')
    
    if aggregate:
        return np.mean(np.array(tot_measures))
    return np.array(tot_measures)


#calcola l'importanza delle features instance per instance come shap
def feature_importance_rtc(model, X_ts, y_true, **kwargs):
    classes = np.unique(y_true)
    if len(classes) > 2:
        importances = model.local_interpretation(X_ts)[2] #se multiclasse è (n inst, n feat, n class)
    else:
        importances = model.local_interpretation(X_ts)[2][:, :, 1] #se bin è (n inst, n feat)
    
    return importances  


#stessa cosa ma per lr
def feature_importance_lr(model, X_ts, y_true, **kwargs):
    #weights (n_classes, n_features), X_ts (n_instances, n_features)
    weights = model.coef_ 
    classes = np.unique(y_true)
    
    if len(classes) > 2:
        unordered_fc = X_ts[:, None, :] * weights[None, :, :]
        feature_contributions = np.transpose(unordered_fc, (0, 2, 1))
        
    else:
        feature_contributions = X_ts * weights
        
    return feature_contributions


#calcola la similarità nelle feature importances specifiche per rtc e lr, date le stesse previsioni    
#model type possibili sono ['rtc', 'lr']    
def feature_importance_similarity(models, X_ts, importance_metric = feature_importance_rtc,
                                      preds_concordance = True, 
                                      metric = spearmanr,
                                      **kwargs
                                      ):
    X_ts = prediction_concordance_filter(models, X_ts, to_filter = X_ts, concordant = preds_concordance)
    
    if importance_metric is feature_importance_rtc:
        imp1 = importance_metric(models[0], X_ts)
        imp2 = importance_metric(models[1], X_ts)
    
    elif importance_metric is feature_importance_lr:
        imp1 = importance_metric(models[0], X_ts)
        imp2 = importance_metric(models[1], X_ts)
    
    return importance_array_similarity(imp1, imp2, metric = metric)


#rende gli shap values per le features di un modello
def get_shaps(model, X_ts, y_true, model_id = None, expl = KernelExplainer, **kwargs):
    X_ts = np.array(X_ts)
    #perché il kernel non è deterministico (stima Monte Carlo)
    np.random.seed(0)
    
    n_instances = X_ts.shape[0]
    n_samples = int(np.sqrt(n_instances))
    index = np.random.choice(n_instances, size = n_samples, replace = False)
    
    n_classes = len(np.unique(y_true))
    if expl is KernelExplainer:
        
        background = X_ts[index]
        if n_classes == 2:
            explainer = expl(model.predict, background)
        else:
            explainer = expl(model.predict_proba, background)
    elif expl is TreeExplainer:
        explainer = expl(model)
    
    elif expl is LinearExplainer:
        explainer = expl(model, masker = X_ts)
    
    shaps = explainer(X_ts).values
    if model_id is not None:     #l'id mi è utile per le matrici di distanza nell'analisi
        return model_id, shaps
    return shaps


#prende un dizionario di modelli per id, calcola gli shaps per ognuno e mi rende un dizionario degli shaps per id
#mi serve per le matrici di distanza nell'analisi
def get_tot_shaps(models_dict, X_ts, y_true, expl = KernelExplainer, n_jobs = -1,
                  save = False, file_path = '.', file_name = '', **kwargs):
    start = time.time()
    results = Parallel(n_jobs = n_jobs)(delayed(get_shaps)(model, X_ts, y_true, model_id, expl, **kwargs)
                                        for model_id, model in models_dict.items())
    end = time.time()
    res = dict(results)
    if save:
        joblib.dump(res, os.path.join(file_path, f'{file_name}.joblib'))
    
    print(f'time taken: {end - start} seconds')
    return res


#utilizzabile se abbiamo due modelli da paragonare. Se ne abbiamo di più, usiamo altre funzioni
#considera la somiglianza dell'influenza delle features instance per instance (anche filtrato per previsioni concord/discord)
#definibile sia con misure di correlazione che di errore
#se aggregate = False, mi rende un'array (n.inst,) o (n.inst, n.classes) in cui per ogni instance, ho un valore di similarità
def shaps_similarity(models, X_ts,
                     explainer = KernelExplainer, 
                     metric = spearmanr, 
                     preds_concordance = None,
                     aggregate = False,
                     **kwargs
                     ): 
    if preds_concordance is not None:
        #filtriamo
        X_ts = prediction_concordance_filter(models, X_ts, X_ts, concordant = preds_concordance)
    
    imp1 = get_shaps(models[0], X_ts, expl = explainer)
    imp2 = get_shaps(models[1], X_ts, expl = explainer)

    result = importance_array_similarity(imp1, imp2, metric = metric, aggregate = aggregate)
    return result
    






