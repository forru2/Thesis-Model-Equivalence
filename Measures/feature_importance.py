# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:57:47 2025

@author: franc
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from support_functions import prediction_concordance_filter, apply_distance_metric
from shap import TreeExplainer, LinearExplainer, KernelExplainer


#calcola la similarità tra due array delle importanze
#la uso sia in shaps_similarity che in feature_importance_similarity_rtc
def importance_array_similarity(arr1, arr2, metric):
    tot_measures = []
    n_classes = arr1.shape[2] if arr1.ndim == 3 else 1
    for i in range(arr1.shape[0]):
        #multiclasse
        if n_classes > 1:

            instance_measures = []
            for c in range(n_classes): 
                measure = apply_distance_metric(arr1[i, :, c], arr2[i, :, c], metric = metric)
                
                instance_measures.append(measure)
            tot_measures.append(instance_measures)
        #binario
        else:
            measure = apply_distance_metric(arr1[i, :], arr2[i, :], metric = metric)
            tot_measures.append(measure)
            
    return np.array(tot_measures)


#calcola l'importanza delle features instance per instance come shap
def feature_importance_rtc(model, X_ts, multiclass = False):
    if multiclass:
        importances = model.local_interpretation(X_ts)[2]
    else:
        importances = model.local_interpretation(X_ts)[2][:, :, 1]
    
    return importances  


#stessa cosa ma per lr
def feature_importance_lr(model, X_ts, multiclass = False):
    #weights (n_classes, n_features), X_ts (n_instances, n_features)
    weights = model.coef_ 
    
    if multiclass:
        unordered_fc = X_ts[:, None, :] * weights[None, :, :]
        feature_contributions = np.transpose(unordered_fc, (0, 2, 1))
        
    else:
        feature_contributions = X_ts * weights
        
    return feature_contributions


#calcola la similarità nele feature importances specifiche per rtc e lr, date le stesse previsioni    
#model type possibili sono ['rtc', 'lr']    
def feature_importance_similarity(models, X_ts, importance_metric = feature_importance_rtc,
                                      preds_concordance = True, 
                                      multiclass = False, 
                                      metric = spearmanr
                                      ):
    X_ts = prediction_concordance_filter(models, X_ts, to_filter = X_ts, concordant = preds_concordance)
    
    if importance_metric is feature_importance_rtc:
        imp1 = importance_metric(models[0], X_ts, multiclass = multiclass)
        imp2 = importance_metric(models[1], X_ts, multiclass = multiclass)
    
    elif importance_metric is feature_importance_lr:
        imp1 = importance_metric(models[0], X_ts, multiclass = multiclass)
        imp2 = importance_metric(models[1], X_ts, multiclass = multiclass)
    
    return importance_array_similarity(imp1, imp2, metric = metric)


#rende gli shap values per le features di un modello
def get_shaps(model, X_ts, expl = KernelExplainer):
    
    #perché il kernel non è deterministico (stima Monte Carlo)
    np.random.seed(0)
    
    n_instances = X_ts.shape[0]
    n_samples = int(np.sqrt(n_instances))
    index = np.random.choice(n_instances, size = n_samples, replace = False)
    
    n_classes = model.predict_proba(X_ts[:1]).shape[1]
    if expl == KernelExplainer:
        
        background = X_ts[index]
        if n_classes == 2:
            explainer = expl(model.predict, background)
        else:
            explainer = expl(model.predict_proba, background)
    elif expl == TreeExplainer:
        explainer = expl(model)
    
    elif expl == LinearExplainer:
        explainer = expl(model, masker = X_ts)
    
    shaps = explainer(X_ts).values
    return shaps


#considera la somiglianza dell'influenza delle features instance per instance in caso di previsioni concord/discord
#definibile sia con misure di correlazione che di errore
def shaps_similarity(models, X_ts,
                     explainer = KernelExplainer, 
                     metric = spearmanr, 
                     preds_concordance = True,
                     min_centroids = 2, 
                     max_centroids = 50, 
                     sampled_results = False):    
    #filtriamo
    X_ts = prediction_concordance_filter(models, X_ts, X_ts, concordant = preds_concordance)
    
    imp1 = get_shaps(models[0], X_ts, expl = explainer)
    imp2 = get_shaps(models[1], X_ts, expl = explainer)
       
    return importance_array_similarity(imp1, imp2, metric = metric)







