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
from preprocess_and_training import get_parameter_combos, model_results, grid
from joblib import Parallel, delayed
from my_knn import knn



    
def features_combos(X_tr, p:list, n_neighbors:list, weights:list, n_total_combos = 1000, random_state = 42, prt = False):
    
    random.seed(random_state)
    
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



def results_knn(X_train, y_train, X_ts, X_val, y_ts, y_val, parameter_combo:dict, metrics_to_compute, 
                    sens_feat_train, sens_feat_ts, sens_feat_val,
                    output_dir='.', model_dir = '.'):
    
   
    return results_manager(model = knn, X_train = X_train, X_ts = X_ts, X_val = X_val, y_train = y_train, y_ts = y_ts, 
                           y_val = y_val, parameter_combo = parameter_combo, 
                           metrics_to_compute = metrics_to_compute, 
                           sens_feat_train = sens_feat_train, sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                           output_dir = output_dir, model_dir = model_dir
                           )
    

    
if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/compass'
    X_train = np.genfromtxt(f'{path}/compass_X_train.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/compass_X_ts.csv', delimiter=',', skip_header=1)
    X_val = np.genfromtxt(f'{path}/compass_X_val.csv', delimiter=',', skip_header=1)
    
    y_ts = np.genfromtxt(f'{path}/compass_y_ts.csv', delimiter=',', skip_header=1)
    y_val = np.genfromtxt(f'{path}/compass_y_val.csv', delimiter=',', skip_header=1)
    y_train = np.genfromtxt(f'{path}/compass_y_train.csv', delimiter=',', skip_header=1)
    
    X_ts_df = pd.read_csv(f'{path}/compass_X_ts.csv')
    X_val_df = pd.read_csv(f'{path}/compass_X_val.csv')
    X_train_df = pd.read_csv(f'{path}/compass_X_train.csv')  
    
    knn_param_grid = {
        'weights': ['uniform', 'distance'],               
        'n_neighbors': [2, 6, 10, 20, 30], 
        'p': [1, 2], 
        'features_knn': features_combos(X_tr = X_train, p = [1, 2], n_neighbors = [2, 6, 10, 20, 30],
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
         X_train = X_train, X_val = X_val, y_train = y_train, X_ts = X_ts, y_ts = y_ts, y_val = y_val, 
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         sens_feat_train = X_train_df['race'].values,
         sens_feat_ts = X_ts_df['race'].values,
         sens_feat_val = X_val_df['race'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\compass\model_results\knn', 
         model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\compass\knn',
        ) for combo in parameter_combos_knn)
    
   
    
    