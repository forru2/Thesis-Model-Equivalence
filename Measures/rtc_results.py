# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 22:48:18 2026

@author: franc
"""

import pandas as pd
import numpy as np
from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
import fairness as fn
import performance as pf
import robustness as rb
from joblib import Parallel, delayed
from preprocess_and_training import results_manager, get_parameter_combos, holdout_results, full_model_results, grid
import model_complexity as mc


def manage_rtc_base_stumps(parameter_combo:dict, rng: np.random.RandomState):
    param_combo = parameter_combo.copy()
    
    param_combo['stump_selection'] = 'random'
    param_combo['base_stumps'] = [(0.6, DecisionTreeStumpClassifier(splitter='random', random_state = rng, max_depth=1)),
                                      (0.4, DecisionTreeStumpClassifier(max_depth=1, splitter='best', random_state = rng))]

    return param_combo



def results_rtc(X_tr, y_tr, X_ts, y_ts, results_function, parameter_combo:dict, metrics_to_compute, 
                    fairness_sens_feat_tr, fairness_sens_feat_ts,
                    output_dir='.', val_size=0.2, random_state = None, model_dir = '.'):
    
    rng = np.random.RandomState(random_state)
    param_combo_processed = manage_rtc_base_stumps(parameter_combo = parameter_combo, rng = rng)
        
    return results_manager(model = RuleTreeClassifier, X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
                           results_function = results_function, parameter_combo = parameter_combo, 
                           metrics_to_compute = metrics_to_compute, 
                           fairness_sens_feat_tr = fairness_sens_feat_tr, fairness_sens_feat_ts = fairness_sens_feat_ts,
                           output_dir = output_dir, val_size = val_size, random_state = random_state, model_dir = model_dir,
                           param_combo_processed = param_combo_processed, add_rs = True
                           )
    



if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    X_ts_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')
    X_tr_df = pd.read_csv(f'{path}/german_credit_X_tr.csv')
    
    rtc_param_grid = {
        'criterion': ['gini', 'entropy'],               
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 
        'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 75, 100, 150], 
        'min_samples_leaf': [1, 2, 5, 10, 20]            
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
    {'name': 'rtc_depth', 'func': mc.rtc_depth, 'params': {}},
    {'name': 'rtc_n_nodes', 'func': mc.count_nodes_rtc, 'params': {}},
    {'name': 'rtc_n_rules', 'func': mc.n_rules, 'params': {}}
    ]
    
    
    parameter_combos_rtc = get_parameter_combos(rtc_param_grid)
    
    Parallel(n_jobs=-1)(delayed(results_rtc)(
         X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
         results_function = holdout_results, 
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
         fairness_sens_feat_ts = X_ts_df['personal_status'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\rtc', 
         val_size=0.2,
         random_state = i
        ) for i, combo in enumerate(parameter_combos_rtc))
    
    Parallel(n_jobs=-1)(delayed(results_rtc)(
         X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
         results_function = full_model_results, 
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
         fairness_sens_feat_ts = X_ts_df['personal_status'].values,
         output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\rtc', 
         random_state = i,
         model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\rtc' 
        ) for i, combo in enumerate(parameter_combos_rtc))
    







