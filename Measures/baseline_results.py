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
from preprocess_and_training import model_results
from rtc_results import results_rtc
from knn_results import results_knn, features_combos
from lr_results import single_lr_results
import fairness as fn
import performance as pf
import robustness as rb
import model_complexity as mc
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
#from sklearn.metrics import classification_report, accuracy_score
#from joblib import load
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_train = np.genfromtxt(f'{path}/german_credit_X_train.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    X_val = np.genfromtxt(f'{path}/german_credit_X_val.csv', delimiter=',', skip_header=1)
    
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_val = np.genfromtxt(f'{path}/german_credit_y_val.csv', delimiter=',', skip_header=1)
    y_train = np.genfromtxt(f'{path}/german_credit_y_train.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    
    X_ts_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')
    X_val_df = pd.read_csv(f'{path}/german_credit_X_val.csv')
    X_train_df = pd.read_csv(f'{path}/german_credit_X_train.csv')  
    
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
    
    
    
    knn_param_grid = {
        'weights': ['uniform', 'distance'],               
        'n_neighbors': [2, 6,  10, 20, 30], 
        'p': [1, 2], 
        'features_knn': features_combos(X_tr = X_tr, p = [1, 2], n_neighbors = [ 2,6, 10, 20, 30],
                                    weights = ['uniform', 'distance'], n_total_combos = 1000)
         }
    
    
    knn_results = grid(X_tr, y_tr, knn_param_grid, model = knn(), n_jobs = 1)
    print(knn_results)
    print('overfitting: ', round(knn_results['train_f1'] - knn_results['val_f1'], 4))
    

    
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
        'ccp_alpha': [0.01],
        #questi due mi servono solo perché vengano utilizzati dentro pars per results manager
        'class_weight': ['balanced_subsample'],
        'n_jobs': [-1]
        }
    
    results = grid(X_tr, y_tr, param_grid = rf_param_grid,
                   model = RandomForestClassifier(random_state = 42))
    
    
    lgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1],   
        'num_leaves': [30, 50, 70, 100],          
        'max_depth': [-1],            
        'min_child_samples': [30, 50, 70, 100],   
        'reg_alpha': [1, 2],   #L1        
        'reg_lambda': [1, 2],  #L2        
        'is_unbalance': [True],          
        'importance_type': ['gain'],
        'n_jobs': [-1]
    }
    
    results = grid(X_tr, y_tr, param_grid = lgb_param_grid,
                   model = LGBMClassifier(random_state = 42))
    
    print(results)
    print('overfitting: ', results['train_f1'] - results['val_f1'])
            
    
    cb_param_grid = {
            'iterations': [100, 200],         
            'learning_rate': [0.01, 0.05],
            'depth': [3, 4],                  
            'l2_leaf_reg': [10, 30, 50],      
            'random_strength': [5, 10],       
            'bagging_temperature': [0.5, 1.0],
            'min_data_in_leaf': [10, 20],     
            'auto_class_weights': ['Balanced']
        }
    
    results = grid(X_tr, y_tr, param_grid = cb_param_grid,
                   model = CatBoostClassifier(random_state = 42))
    
    print(results)
    print('overfitting: ', results['train_f1'] - results['val_f1'])
        
    
    xgb_param_grid = {
            'max_depth': [3, 4, 5],             
            'n_estimators': [100, 150, 200],    
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [2.5, 3, 3.5],   
            'reg_lambda': [3, 10, 50],         
            'reg_alpha': [3, 10],            
            'gamma': [3, 5],           
            'subsample': [0.5, 0.7, 1.0],      
            'colsample_bytree': [0.5, 0.7, 1.0] 
        }
    
    results = grid(X_tr, y_tr, param_grid = xgb_param_grid,
                   model = XGBClassifier(random_state = 42))
    
    print(results)
    print('overfitting: ', results['train_f1'] - results['val_f1'])
        
    pars = results['best_params']   
    baseline = results_manager(model = XGBClassifier,
                               X_train = X_train, X_val = X_val, X_ts = X_ts,
                               y_train = y_train, y_val = y_val, y_ts = y_ts, 
                               parameter_combo = pars, 
                               metrics_to_compute = metrics_to_compute, 
                               sens_feat_train = X_train_df['personal_status'].values,
                               sens_feat_ts = X_ts_df['personal_status'].values,
                               sens_feat_val = X_val_df['personal_status'].values,
                               random_state = 42,
                               output_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\baseline',
                               model_dir = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\baseline',
                              )
 
    