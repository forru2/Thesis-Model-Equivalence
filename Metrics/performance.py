# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:22:06 2025

@author: franc
"""
import inspect
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, hamming_loss, recall_score, precision_score, mean_squared_error, mean_absolute_error
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
    


#calcola la similarità/distanza tra le previsioni dei modelli
#usa la jaccard o hamming loss
def predictions_similarity(preds1, preds2, metric = hamming_loss, average = None, **kwargs):

    #preds, _ = ut.predictions(models, X_ts)
    n_classes = len(np.unique(preds1))
    
    if metric is jaccard_score:
        #anche qua con il parametro average scegliamo la jaccard media o per classe
        if n_classes > 2:  
            result = metric(preds1, preds2, average = average)
        else:
            result = metric(preds1, preds2, average = 'binary')
    
    elif metric is hamming_loss:
        result = metric(preds1, preds2)
    else:
        result = metric(preds1, preds2, **kwargs)
        
    return result 
    

#calcola la similarità tra distribuzioni di probabilità predittive in termini di correlazione ed errore 
#se concordano/discordano o meno, quanto sono simili le rispettive distribuzioni di probabilità?   
def probabilities_similarity(models, X_ts, y_true, metric = spearmanr, preds_concordance = None, **kwargs):

    _, probs = ut.predictions(models, X_ts) 
    probs1, probs2 = probs
    
    if preds_concordance is not None:
        #filtriamo per predizioni concordanti o discordanti
        m1_probs, m2_probs = ut.prediction_concordance_filter(models, X_ts, (probs1, probs2), concordant = preds_concordance)
    else:
        m1_probs, m2_probs = probs1, probs2
        
    results = []
    for class_ in range(m1_probs.shape[1]):
        result = ut.apply_distance_metric(m1_probs[:, class_], m2_probs[:, class_], metric)
        
        results.append(result)
    return np.array(results) #similarity per classe
        
    
#se entrambi hanno torto/ragione (o a prescindere), quanto sono convinti della ground truth?
#rende un'array (n.instances, 2) una colonna di confidence per modello se in input ha una lista di due modelli
#rende l'array delle probs sulla gt se in input ha un solo modello
#concordant può essere True/False/None
def true_class_probability(models, X_ts, y_true, concordant = None, **kwargs):

    _, probs = ut.predictions(models, X_ts)
    if isinstance(models, (list, tuple)) and len(models) > 1:
    
        if concordant is not None:
            probs[0], probs[1], y_true = ut.prediction_concordance_filter(models, X_ts, to_filter = (probs[0], probs[1], y_true), y_true = y_true, concordant = concordant)
        
        indexes = y_true.ravel().astype(int)
        probs_true_class = [prob[np.arange(prob.shape[0]), indexes] for prob in probs]
        return np.column_stack((probs_true_class[0], probs_true_class[1])) 
    else:
        probs = (probs)
        indexes = y_true.ravel().astype(int)
        prob_true_class = probs[np.arange(probs.shape[0]), indexes]        
        return prob_true_class 


#calcola l'errore di calibrazione atteso del modello
#sia per cl binaria che multiclasse perché considera a prescindere la max confidence
def expected_calibration_error(model, X_ts, y_true, n_bins = 5):
    y_pred, probs = ut.predictions(model, X_ts)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    max_conf = np.max(probs, axis = 1)
    accuracies = y_pred == y_true
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(max_conf > bin_lower.item(), max_conf <= bin_upper.item())
        prob_in_bin = in_bin.mean() #prob che la confidence sia nel bin

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_conf_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prob_in_bin
    return ece



#calcola la sicurezza media sulla ground truth
def mean_confidence_on_ground_truth(model, X_ts, y_true):
    probs_of_gt = true_class_probability(model, X_ts, y_true)
    return np.mean(probs_of_gt)


#calcola la media dei logaritmi della sicurezza sulla gt
#è più sensibile della sicurezza media rispetto alle probabilità molto basse: ne tiene conto
def cat_cross_entropy_on_ground_truth(model, X_ts, y_true):
    probs_of_gt = true_class_probability(model, X_ts, y_true)
    neg_logs = - np.log(probs_of_gt + 1e-15) #per evitare log(0)
    return np.mean(neg_logs)

#calcola la distanza vettoriale o correlazione tra i vettori delle confidence sulla ground truth
#rende uno scalare, che sarebbe la similarità/distanza
def true_class_prob_similarity(models, X_ts, y_true, metric, concordant = False, **kwargs):
    true_class_probs = true_class_probability(models, X_ts, y_true, concordant = concordant)
    
    p1 = true_class_probs[:, 0]
    p2 = true_class_probs[:, 1]
    
    result = ut.apply_distance_metric(p1, p2, metric)
    return result











