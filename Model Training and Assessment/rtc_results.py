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
from preprocess_and_training import results_manager, get_parameter_combos, model_results, grid
import model_complexity as mc
import reliability as rel


def manage_rtc_base_stumps(parameter_combo:dict, rng: np.random.RandomState):
    param_combo = parameter_combo.copy()
    
    param_combo['stump_selection'] = 'random'
    param_combo['base_stumps'] = [(0.6, DecisionTreeStumpClassifier(splitter='random', random_state = rng, max_depth=1)),
                                      (0.4, DecisionTreeStumpClassifier(max_depth=1, splitter='best', random_state = rng))]

    return param_combo



def results_rtc(X_train, X_val, y_train, X_ts, y_ts, y_val, parameter_combo:dict, metrics_to_compute, 
                    sens_feat_train, sens_feat_ts, sens_feat_val,
                    output_dir='.', random_state = None, model_dir = '.'):
    
    rng = np.random.RandomState(random_state)
    param_combo_processed = manage_rtc_base_stumps(parameter_combo = parameter_combo, rng = rng)
        
    return results_manager(model = RuleTreeClassifier, X_train = X_train, X_val = X_val, 
                           y_train = y_train, X_ts = X_ts, y_ts = y_ts, y_val = y_val,
                           parameter_combo = parameter_combo, 
                           metrics_to_compute = metrics_to_compute, 
                           sens_feat_train = sens_feat_train, sens_feat_ts = sens_feat_ts, sens_feat_val = sens_feat_val,
                           output_dir = output_dir, random_state = random_state, model_dir = model_dir,
                           param_combo_processed = param_combo_processed, add_rs = True
                           )
    



if __name__ == '__main__':
    path = '/home/forru/Split_salvati/adult'
    name = 'adult'
    X_train = np.genfromtxt(f'{path}/{name}_X_train.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/{name}_X_ts.csv', delimiter=',', skip_header=1)
    X_val = np.genfromtxt(f'{path}/{name}_X_val.csv', delimiter=',', skip_header=1)
    
    y_ts = np.genfromtxt(f'{path}/{name}_y_ts.csv', delimiter=',', skip_header=1)
    y_val = np.genfromtxt(f'{path}/{name}_y_val.csv', delimiter=',', skip_header=1)
    y_train = np.genfromtxt(f'{path}/{name}_y_train.csv', delimiter=',', skip_header=1)
    
    X_ts_df = pd.read_csv(f'{path}/{name}_X_ts.csv')
    X_val_df = pd.read_csv(f'{path}/{name}_X_val.csv')
    X_train_df = pd.read_csv(f'{path}/{name}_X_train.csv') 
    
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
    {'name': 'rtc_n_rules', 'func': mc.n_rules, 'params': {}},
    {'name': 'expected_calibration_error', 'func': rel.expected_calibration_error, 'params': {'n_bins': 10}},
    {'name': 'cat_cross_entropy_on_gt', 'func': rel.cat_cross_entropy_on_ground_truth, 'params': {}},
    {'name': 'mean_confidence_on_gt', 'func': rel.mean_confidence_on_ground_truth, 'params': {}}
    ]
    
    
    parameter_combos_rtc = get_parameter_combos(rtc_param_grid)
    
    Parallel(n_jobs=-1)(delayed(results_rtc)(
         X_train = X_train, X_val = X_val, y_train = y_train, X_ts = X_ts, y_ts = y_ts, y_val = y_val,
         parameter_combo = combo, metrics_to_compute = metrics_to_compute, 
         sens_feat_train = X_train_df['sex'].values, 
         sens_feat_ts = X_ts_df['sex'].values,
         sens_feat_val = X_val_df['sex'].values,
         output_dir = '/home/forru/Results/adult/model_results/rtc', 
         model_dir = '/home/forru/Models/adult/rtc',
         random_state = i
        ) for i, combo in enumerate(parameter_combos_rtc))
    
    






