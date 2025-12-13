# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:22:06 2025

@author: franc
"""
import inspect
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, recall_score, precision_score, mean_squared_error, mean_absolute_error
import Utils as ut
from scipy.stats import kendalltau, spearmanr, pearsonr
import numpy as np

#calcolo performance di un modello con le principali misure in sklearn
def performance(model, X_ts, y_true, measure = accuracy_score, average = 'weighted', **kwargs):
  
    y_pred = model.predict(X_ts)
    n_classes = len(np.unique(y_true))
    
    #la misura accuracy_score non ha average come parametro perché agisce in generale e non classe per classe
    supports_average = 'average' in inspect.signature(measure).parameters
    if supports_average:
        
        #se è multiclasse, decidiamo se vogliamo la metrica media o per classe
        #lo facciamo con il parametro average
        if n_classes > 2:
            avg = average 
        else:
            avg = 'binary'
        
        score = measure(y_true, y_pred, average = avg)
    else:
        score = measure(y_true, y_pred)
    
    return score



#calcolo della differenza in performance tra due modelli
def performance_difference(models:list, X_ts, y_true, measure = accuracy_score, average = None, **kwargs):
        
        scores = [performance(model, X_ts, y_true, measure = measure, average = average) for model in models]
        
        if len(scores) > 2:
            raise ValueError(f'The function handles at most two models, you are comparing {len(models)} models')
        
        return abs(scores[0] - scores[1])
    


#calcola la jaccard tra le predizioni dei modelli per definirne la somiglianza
def predictions_similarity(models, X_ts, y_true, average = None, **kwargs):

    preds, _ = ut.predictions(models, X_ts)
    n_classes = len(np.unique(y_true))
    
    #anche qua con il parametro average scegliamo la jaccard media o per classe
    if n_classes > 2:  
        similarity = jaccard_score(preds[0], preds[1], average = average)
    else:
        similarity = jaccard_score(preds[0], preds[1], average = 'binary')
        
    return similarity    
    

#calcola la similarità tra confidence in termini di correlazione ed errore 
#se concordano/discordano, quanto sono simili in sicurezza?   
def confidence_similarity(models, X_ts, y_true, metric = spearmanr, preds_concordance = True, **kwargs):

    _, probs = ut.predictions(models, X_ts) 
    probs1, probs2 = probs
    
    #filtriamo per predizioni concordanti o discordanti
    m1_probs, m2_probs = ut.prediction_concordance_filter(models, X_ts, (probs1, probs2), concordant = preds_concordance)
    #in caso di classificazione binaria
    if m1_probs.shape[1] == 2:
        
        result = ut.apply_distance_metric(m1_probs[:, 1], m2_probs[:, 1], metric)       
        return result
    
    #in caso multiclasse
    else:
        results = []
        for class_ in range(m1_probs.shape[1]):
            result = ut.apply_distance_metric(m1_probs[:, class_], m2_probs[:, class_], metric)
            
            results.append(result)
        return np.array(results)
        
    
#se entrambi hanno torto/ragione, quanto sono convinti della ground truth?
def true_class_probability(models, X_ts, y_true, concordant = False, **kwargs):

    _, probs = ut.predictions(models, X_ts)
    probs1, probs2 = probs
    probs1, probs2, y_true = ut.prediction_concordance_filter(models, X_ts, to_filter = (probs1, probs2, y_true), y_true = y_true, concordant = concordant)
    filtered_probs = probs1, probs2
    
    indexes = y_true.ravel().astype(int)
    probs_true_class = [prob[np.arange(prob.shape[0]), indexes] for prob in filtered_probs]
    
    return np.column_stack((probs_true_class[0], probs_true_class[1]))    
    
#la similarità possiamo calcolarla con apply_distance_metric    











