# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 22:30:50 2026

@author: franc
"""

import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression 
import fairness as fn
import performance as pf
import robustness as rb
from joblib import Parallel, delayed
from preprocess_and_training import results_manager, get_parameter_combos, model_results
import model_complexity as mc
import random
from my_lr import lr

def features_combos_lr(X_tr, penalty:list, C:list, solver = 'saga', max_iter = 100000,
                       n_total_combos = 1000, prt = False):
    
    random.seed(0)
    
    n_features = X_tr.shape[1]
    features_indices = list(range(n_features))
    n_base_combos = len(penalty)*len(C)
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


def add_noise_lr(model, seed, std):
    if hasattr(model, 'coef_'):
        rng = np.random.RandomState(seed)
        model.coef_ += rng.normal(loc = 0, scale = std * np.abs(model.coef_))
        model.intercept_ += rng.normal(loc = 0, scale = std * np.abs(model.intercept_))
        
    return model

def single_lr_results(X_train, y_train, X_ts, X_val, y_ts, y_val, parameter_combo:dict, metrics_to_compute, 
                    sens_feat_train, sens_feat_ts, sens_feat_val, std = 0.0,
                    output_dir='.', random_state = None, model_dir = '.'):
    
    apply_noise = lambda m, rand_state: add_noise_lr(m, rand_state, std)
    
    lr_param_combo = {**parameter_combo, 'noise_std': std}
    
    return results_manager(model = lr, X_train = X_train, y_train = y_train, X_ts = X_ts, X_val = X_val, y_ts = y_ts, 
                           y_val = y_val,
                           parameter_combo = lr_param_combo, 
                           param_combo_processed = parameter_combo,
                           metrics_to_compute = metrics_to_compute, 
                           sens_feat_train = sens_feat_train, sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                           output_dir = output_dir, random_state = random_state, model_dir = model_dir,
                           add_rs = True, noise_function = apply_noise
                           )


def results_lr(combo, X_train, y_train, X_ts, X_val, y_ts, y_val, metrics_to_compute, sens_feat_train,
               sens_feat_ts, sens_feat_val, output_dir, model_dir = '.', n_noisy_versions = 3, std = 0.1, seed = 0):
    

    single_lr_results(X_train = X_train, y_train = y_train, X_ts = X_ts, X_val = X_val, y_ts = y_ts, y_val = y_val, 
                      parameter_combo = combo, metrics_to_compute = metrics_to_compute,
                      sens_feat_train = sens_feat_train,
                      sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                      std = 0.0, output_dir = output_dir, random_state = seed, model_dir = model_dir)
        
    for n in range(n_noisy_versions):
        seed += 1 
        single_lr_results(X_train = X_train, y_train = y_train, X_ts = X_ts, X_val = X_val, y_ts = y_ts, y_val = y_val, 
                          parameter_combo = combo, metrics_to_compute = metrics_to_compute,
                          sens_feat_train = sens_feat_train,
                          sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                          std = std, output_dir = output_dir, random_state = seed, model_dir = model_dir)
 
            
    
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
    
    lr_param_grid = {       
        'solver': ['saga'],
        'penalty': ['l1', 'l2'],   
        'C': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200],
        'max_iter': [100000],
        'features_lr': features_combos_lr(X_train, penalty = ['l1', 'l2'],
                                       C = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200], 
                                       n_total_combos = 520)
        }
    
    
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
    {'name': 'eq_odds_robustness', 'func': rb.robustness, 'params': {'measure': fn.advanced_fairness, 'measure_metric': 'cuae'}},
    {'name': 'coeff_magnitude', 'func': mc.lr_complexity, 'params': {'p': 2, 'average': True}}
    ]
    
    parameter_combos_lr = get_parameter_combos(lr_param_grid)

    Parallel(n_jobs=-1)(delayed(results_lr)(
         X_train = X_train, y_train = y_train, X_ts = X_ts, X_val = X_val, y_ts = y_ts, y_val = y_val, 
         combo = combo, metrics_to_compute = metrics_to_compute, 
         sens_feat_train = X_train_df['race'].values, 
         sens_feat_ts = X_ts_df['race'].values,
         sens_feat_val = X_val_df['race'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\compass\model_results\lr', 
         model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\compass\lr',
         n_noisy_versions = 1, std = 0.5 , seed = i
        ) for i, combo in enumerate(parameter_combos_lr))
    
    

