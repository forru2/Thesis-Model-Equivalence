# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:39:03 2026

@author: franc
"""

from preprocess_and_training import grid, results_manager
from RuleTree import RuleTreeClassifier
from sklearn.linear_model import LogisticRegression 
from my_knn import knn
import numpy as np
import pandas as pd
from preprocess_and_training import full_model_results, holdout_results
from rtc_results import results_rtc
from knn_results import results_knn, features_combos
from lr_results import single_lr_results
import fairness as fn
import performance as pf
import robustness as rb
import model_complexity as mc
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from joblib import load


if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    X_ts_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')
    X_tr_df = pd.read_csv(f'{path}/german_credit_X_tr.csv')
    
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
    
    
    
    
    rtc_param_grid = {
        'criterion': ['gini', 'entropy'],               
        'max_depth': [3, 4], 
        'min_samples_split': [30], 
        'min_samples_leaf': [30]            
         }
    
    rtc_results = grid(X_tr, y_tr, rtc_param_grid, model = RuleTreeClassifier)
    print(rtc_results)
    print('overfitting: ', round(rtc_results['train_f1'] - rtc_results['val_f1'], 4))
    rtc_pars = rtc_results['best_params']
    rtc = results_rtc(X_tr, y_tr, X_ts, y_ts, 
                      results_function = holdout_results, 
                      parameter_combo = rtc_pars, 
                      metrics_to_compute = metrics_to_compute, 
                      fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
                      fairness_sens_feat_ts = X_ts_df['personal_status'].values,
                      output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\baseline\rtc_complexity', 
                      random_state = 42,
                      model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\baseline\rtc_complexity' 
                      )
    
    
    knn_param_grid = {
        'weights': ['uniform', 'distance'],               
        'n_neighbors': [6, 9, 16], 
        'p': [1, 2], 
        'features': features_combos(X_tr = X_tr, p = [1, 2], n_neighbors = [6, 9, 16],
                                    weights = ['uniform', 'distance'], n_total_combos = 15)
         }
    
    
    knn_results = grid(X_tr, y_tr, knn_param_grid, model = knn, n_jobs = 1)
    print(knn_results)
    print('overfitting: ', round(knn_results['train_f1'] - knn_results['val_f1'], 4))
    
    knn_pars = knn_results['best_params']
    knn_ = results_knn(X_tr, y_tr, X_ts, y_ts, 
                      results_function = holdout_results, 
                      parameter_combo = knn_pars, 
                      metrics_to_compute = metrics_to_compute, 
                      fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
                      fairness_sens_feat_ts = X_ts_df['personal_status'].values,
                      output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\baseline\knn_complexity', 
                      model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\baseline\knn_complexity' 
                      )
    
    lr_param_grid = { 
        'solver': ['saga'], #per accertarmi che sia usato per elasticnet             
        'penalty': ['l1', 'l2', 'elasticnet'],   
        'C': [100, 200],
        'max_iter': [100000],
        'l1_ratio': [0.2, 0.5, 0.8] #per accertarmi che sia definito per elasticnet
        }
    
    lr_results = grid(X_tr, y_tr, lr_param_grid, model = LogisticRegression)
    print(lr_results)
    print('overfitting: ', round(lr_results['train_accuracy'] - lr_results['val_accuracy'], 4))
    
    lr_pars = lr_results['best_params']
    lr = single_lr_results(X_tr, y_tr, X_ts, y_ts, 
                           results_function = full_model_results, 
                           parameter_combo = lr_pars, 
                           metrics_to_compute = metrics_to_compute, 
                           fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
                           fairness_sens_feat_ts = X_ts_df['personal_status'].values,
                           std = 0.0,
                           output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\baseline\lr_complexity', 
                           random_state = 42,
                           model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\baseline\lr_complexity'                           
                           )
    
    
    
    xgb_param_grid = {
        'n_estimators': [20, 25],
        'learning_rate': np.linspace(0.3, 0.5, 3),
        'max_depth': [1, 2],
        'gamma': np.linspace(0.6, 0.9, 3),
        'subsample': np.linspace(0.5, 0.8, 2),
        'min_child_weight': [2, 3],
        'reg_alpha': np.linspace(0.1, 0.8, 3), #l1
        'reg_lambda': np.linspace(1, 2, 3) #l2
        }
    
    xgb_results = grid(X_tr, y_tr, xgb_param_grid, model = xgb.XGBClassifier)
    
    print(xgb_results)
    print('overfitting: ', round(xgb_results['train_f1'] - xgb_results['val_f1'], 4))
    
    xgb_pars = xgb_results['best_params']
    
    xgboost = results_manager(model = xgb.XGBClassifier, X_tr = X_tr, y_tr = y_tr, X_ts = X_ts, y_ts = y_ts, 
                              results_function = holdout_results, 
                              parameter_combo = xgb_pars, 
                              metrics_to_compute = metrics_to_compute, 
                              fairness_sens_feat_tr = X_tr_df['personal_status'].values, 
                              fairness_sens_feat_ts = X_ts_df['personal_status'].values,
                              output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\baseline\xgboost',
                              model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\baseline\xgboost',
                              )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    