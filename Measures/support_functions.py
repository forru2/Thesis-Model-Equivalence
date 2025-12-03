# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 13:11:57 2025

@author: franc
"""

import pandas as pd
import numpy as np
from RuleTree import RuleTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, confusion_matrix, multilabel_confusion_matrix
from scipy.stats import kendalltau, spearmanr, pearsonr
import random as rd
from joblib import dump, load
from shap import TreeExplainer, LinearExplainer, KernelExplainer
#import os
import inspect
from fairlearn.metrics import selection_rate, MetricFrame, true_positive_rate, false_positive_rate
from kneed import KneeLocator 


#prende un modello o una lista di modelli e ne calcola le predizioni e le probabilità per ogni classe
def predictions(models, X_ts, limit_to_two = True):
        
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

#applica una metrica facendo la differenza tra quelle che rendono tuple e quelle
#che rendono un valore
def apply_distance_metric(x, y, metric): 
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

def prediction_concordance_filter(models, X_ts, to_filter, y_true = None, concordant = True):
    
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

    













if __name__=='__main__':
    #print(help(ClassificationMetric))
    df_name, df = read_german_credit(basepath = "C:/Users/franc/OneDrive/Magistrale/Tesi modelli equivalenti/")
    
    #X_tr, X_ts, y_tr, y_ts = split_ts(df_name, df, save_path = 'C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Split_salvati')
    #model_types = ['knn', 'rtc', 'lr']
    #for model in model_types:
        #train_models(model_type= model, 
                     #df_name = df_name, 
                     #X_tr = X_tr, 
                     #X_ts = X_ts, 
                     #y_tr = y_tr, 
                     #y_ts = y_ts, 
                     #save_path = 'C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Models/german'
                     #)
    #binary
    X_tr = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_true = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    m1 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_lr_3.joblib')
    m2 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_lr_3.joblib')
    
    models = [m1, m2]
    X_ts_df = pd.DataFrame(X_ts, columns=df.columns[:-1])
    sensitive_features = X_ts_df['personal_status']
    
    #multiclass
    df_name, df = read_vehicle(basepath = "C:/Users/franc/OneDrive/Magistrale/Tesi modelli equivalenti/")
    
    X_tr = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_X_ts.csv', delimiter=',', skip_header=1)
    y_true = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_y_ts.csv', delimiter=',', skip_header=1)
    m1 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/vehicle/vehicle_rtc_1.joblib')
    m2 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/vehicle/vehicle_rtc_1.joblib')
   
    models = [m1, m2]
    X_ts_df = pd.DataFrame(X_ts, columns=df.columns[:-1])
    sensitive_features = X_ts_df['CIRCULARITY']
    sensitive_features = pd.qcut(X_ts_df['CIRCULARITY'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    
    ex = fairness_explorer(m1, X_ts, y_true, sensitive_features, fairness_metrics)
    print(ex)
    for el in ex:
        print(el)
        
    af = advanced_fairness(m1, X_ts, y_true, sensitive_features)
    print(af)
    fairness_metrics = {
        'demographic parity(selection_rate)': selection_rate,
        'equal opportunity(TPR)': true_positive_rate,
        'predictive equality(FPR)': false_positive_rate,
        'predictive parity(precision)': precision_score,
        'negative predictive parity(NPV)': negative_predictive_value}

    
    
    #corretta
    preds, probs = predictions(models, X_ts)
    print(preds[0])
    
    #corretta
    score = performance(models[0], X_ts, y_true, 
                        measure = f1_score,
                        multiclass = True,
                        average = None)
    print(score)
    
    #corretta
    perf_diff = performance_difference(models, X_ts, y_true,
                                       measure = f1_score,
                                       multiclass = True,
                                       average = 'weighted')
    assert perf_diff == 0, 'le performance dello stesso modello devono essere uguali'
    print(perf_diff)
    
    
    #corretto
    similarity = predictions_similarity(models, X_ts,
                                        average = 'weighted',
                                        multiclass = True)
    assert similarity == 1, 'le performance dello stesso modello devono essere uguali'
    print(similarity)

    print(confidence_similarity(models, X_ts, y_true))
    assert confidence_similarity(models, X_ts, y_true) == 1
    
    imp1 = get_shaps(m1, X_ts)
    imp2 = get_shaps(m2, X_ts)
    importance_array_similarity(imp1, imp2, metric= spearmanr)
    
    x = shaps_similarity(models, X_ts, sampled_results=True)
    print(x)
    
    centr = kmeans_centroids(X_ts, min_centroids = 2, max_centroids = 50)
    print(centr.shape)

    shaps = get_shaps(m1, X_ts, expl = TreeExplainer, sampled_results = True)
    print(shaps.shape)
    print(shaps[1, :, 1])
    
    
    fairness = model_fairness(m1, X_ts, y_true, sensitive_features = X_ts_df['purpose_education'])
    print(fairness)
    explorer = fairness_explorer(m1, X_ts, y_true, sensitive_features = X_ts_df['purpose_education'])
    print(explorer)
    
    fairness_metrics = {
                        'demographic_parity_diff': demographic_parity_difference,
                        'equalized_odds_diff': equalized_odds_difference,
                        'equal_opportunity_diff': equal_opportunity_difference
                        }

    fairness = model_fairness_collection(m1,
                                         X_ts,
                                         y_true, 
                                         sensitive_features = X_ts_df['purpose_education'], 
                                         metrics = fairness_metrics,
                                         return_names = False)
    print(type(fairness))


    rules = m2.get_rules(columns_names=df.columns)
    rules
    print(rules.keys())
    print(rules['feature_idx'])
    print(rules['feature_name'], rules['threshold'])
    print(rules['left_node']['left_node']['right_node']['node_id'])
    rules = m2.get_rules(columns_names=df.columns)
    
    tree_depth(rules)
    count_nodes_rtc(rules)
    print(tree_complexity(m2, rules))
    
    m2.print_rules(rules, columns_names=df.columns)
    m2.get_leaf_nodes()

    help(RuleTreeClassifier())
    print(inspect.getsource(RuleTreeClassifier._predict))
    
    for el in m2.local_interpretation(X_ts):
        print(el.shape)
    
    m2.local_interpretation(X_ts)[2][1]
    len(m2.local_interpretation(X_ts))

    imp = feature_importance_rtc(m2, X_ts, feature_names = df.columns[:-1])
    print(imp)
    imp.shape
    
    shap = get_shaps(m1, X_ts)
    print(shap)


    sim = shaps_similarity(models, X_ts, metric = pearsonr)
    sim2 = feature_importance_similarity(models, X_ts, model_type = 'rtc', multiclass = True)
    sim3 = feature_importance_similarity(models, X_ts, model_type = 'lr', multiclass = True)   
    print(sim3.shape)
    
    print(sim2)
    sim.shape
    sim2.shape
    imp = feature_importance_rtc(m2, X_ts, multiclass = True)
    imp.shape
    
    coef = m2.coef_
    print(coef)
    print(coef.shape)
    
    fi = feature_importance_lr(m2, X_ts, multiclass = True) 
    print(fi.shape)      
        
    compl = lr_complexity(m1, multiclass = True)
    print(compl)   

    idx = m2.apply(X_ts)
    print(range(idx.shape[0]))
    lengths = path_lengths(m2, X_ts)
    print(lengths)

    choose_knee_centroids(X_ts, min_centroids=2, max_centroids=50)

    out = true_class_probability(models, X_ts, y_true)
    print(out)
    print(out.shape)












