# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 19:11:03 2025

@author: franc
"""

import pandas as pd
import random
from preprocess_and_training import results_manager
import numpy as np
import fairness as fn
import performance as pf
import robustness as rb
from preprocess_and_training import get_parameter_combos, holdout_results, full_model_results, grid
from joblib import Parallel, delayed
from my_knn import knn



    
def features_combos(X_tr, p:list, n_neighbors:list, weights:list, n_total_combos = 1000, prt = False):
    
    random.seed(0)
    
    n_features = X_tr.shape[1]
    features_indices = list(range(n_features))
    n_base_combos = len(p)*len(weights)*len(n_neighbors)
    n_features_combos = n_total_combos // n_base_combos
    
    features_combos = []
    seen_subsets = set()
    
    n_iter = 0
    while len(features_combos) < n_features_combos and n_iter < n_features_combos * 100:
        feat_subset_size = random.randint(2, n_features)
        feat_subset = random.sample(features_indices, feat_subset_size)
        
        feat_subset_tuple = tuple(sorted(feat_subset))
        if feat_subset_tuple not in seen_subsets:
            seen_subsets.add(feat_subset_tuple)
            features_combos.append(list(feat_subset_tuple))
        n_iter +=1 
    
    if prt:
        print(f'n_features_combos: {len(features_combos)}')
        print(f'tot_combos: {len(features_combos)* n_base_combos}')
            
    return features_combos
    
    
def results_knn(X_tr, y_tr, X_ts, y_ts, results_function, parameter_combo:dict, metrics_to_compute, 
                    fairness_sens_feat_tr, fairness_sens_feat_ts,
                    output_dir='.', val_size=0.2, model_dir = '.'):
    
   
    return results_manager(model = knn, X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
                           results_function = results_function, parameter_combo = parameter_combo, 
                           metrics_to_compute = metrics_to_compute, 
                           fairness_sens_feat_tr = fairness_sens_feat_tr, fairness_sens_feat_ts = fairness_sens_feat_ts,
                           output_dir = output_dir, val_size = val_size, model_dir = model_dir
                           )
    

    
if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    X_ts_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')
    X_tr_df = pd.read_csv(f'{path}/german_credit_X_tr.csv')    
    
    knn_param_grid = {
        'weights': ['uniform', 'distance'],               
        'n_neighbors': [3, 6, 9, 12, 16], 
        'p': [1, 2], 
        'features': features_combos(X_tr = X_tr, p = [1, 2], n_neighbors = [3, 6, 9, 12, 16],
                                    weights = ['uniform', 'distance'], n_total_combos = 1000)
         }
    
    
    parameter_combos_knn = get_parameter_combos(knn_param_grid)
    
    metrics_to_compute = [
    {'name': 'accuracy', 'func': pf.performance, 'params': {'measure': pf.accuracy_score, 'average': 'weighted'}},
    {'name': 'f1', 'func': pf.performance, 'params': {'measure': pf.f1_score, 'average': 'weighted'}},
    {'name': 'precision', 'func': pf.performance, 'params': {'measure': pf.precision_score, 'average': 'weighted'}},
    {'name': 'recall', 'func': pf.performance, 'params': {'measure': pf.recall_score, 'average': 'weighted'}},
    {'name': 'equalized_odds', 'func': fn.advanced_fairness, 'params': {'metric': 'eo'}},
    {'name': 'cuae', 'func': fn.advanced_fairness, 'params': {'metric': 'cuae'}},
    {'name': 'dem_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.selection_rate}},
    {'name': 'equal_opportunity', 'func': fn.fairness_explorer, 'params': {'metric': fn.true_positive_rate}},
    {'name': 'predictive_equality', 'func': fn.fairness_explorer, 'params': {'metric': fn.false_positive_rate}},
    {'name': 'predictive_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.precision_score}},
    {'name': 'neg_predictive_parity', 'func': fn.fairness_explorer, 'params': {'metric': fn.negative_predictive_value}},
    {'name': 'acc_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.accuracy_score}},
    {'name': 'f1_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.f1_score}},
    {'name': 'precision_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.precision_score}},
    {'name': 'recall_robustness', 'func': rb.robustness, 'params': {'measure': rb.performance, 'measure_metric': rb.recall_score}},
    {'name': 'dem_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.selection_rate}},
    {'name': 'equal_opp_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.true_positive_rate}},
    {'name': 'predictive_eq_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.false_positive_rate}},
    {'name': 'pred_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.precision_score}},
    {'name': 'neg_pred_parity_robustness', 'func': rb.robustness, 'params': {'measure': fn.fairness_explorer, 'measure_metric': fn.negative_predictive_value}},
    {'name': 'eq_odds_robustness', 'func': rb.robustness, 'params': {'measure': fn.advanced_fairness, 'measure_metric': 'eo'}},
    {'name': 'eq_odds_robustness', 'func': rb.robustness, 'params': {'measure': fn.advanced_fairness, 'measure_metric': 'cuae'}}
    ]
    
    Parallel(n_jobs=-1)(delayed(results_knn)(
         X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
         results_function = holdout_results, 
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
         fairness_sens_feat_ts = X_ts_df['personal_status'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\knn', 
         val_size=0.2, 
        ) for combo in parameter_combos_knn)
    
    Parallel(n_jobs=-1)(delayed(results_knn)(
         X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
         results_function = full_model_results, 
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
         fairness_sens_feat_ts = X_ts_df['personal_status'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\knn', 
         model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\knn' 
        ) for combo in parameter_combos_knn)
    
    
    