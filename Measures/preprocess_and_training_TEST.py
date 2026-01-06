# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:28:07 2025

@author: franc
"""
import pandas as pd
import numpy as np
from HybridReaders import read_wdbc, read_compass, read_german_credit, read_vehicle
import preprocess_and_training as pt
from joblib import load
from RuleTree import RuleTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.linear_model import LogisticRegression 
import performance as pf
import fairness as fn
import robustness as rb
from sklearn.model_selection import train_test_split




def test_split_data(def_name, df, basepath, X_tr_df, X_tr, scaler, feature_names):
    print('HEAD: \n', X_tr_df.head())
    print('COLUMNS: \n', X_tr_df.columns)
    print('SCALER: ', scaler)
    
    initial_cols = df.columns[:-1].tolist()
    assert initial_cols == feature_names, 'Error: the feature names do not coincide'
    assert isinstance(X_tr, np.ndarray), 'Error: X_tr is not a np array'
    assert X_tr_df.shape == X_tr.shape, 'Error: the X_tr and its csv dimensions do not coincide'



def test_get_metrics(X_tr, X_ts, y_tr, y_ts, metrics_to_compute, sens_feat_tr, sens_feat_ts):
    model = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\knn\00a6c1a79809de26f3745c8f15977157.joblib')
    param_combo = model.get_params()  
    mtrc1 = pt.get_metrics(model, X_tr, y_tr, X_ts, y_ts, metrics_to_compute, sens_feat_tr, sens_feat_ts, 
                           param_combo, prefix = 'val_')
    
    assert isinstance(mtrc1, dict)
    print(mtrc1.keys())
    mtrc2 = pt.get_metrics(model, X_tr, y_tr, X_ts, y_ts, metrics_to_compute, sens_feat_tr, sens_feat_ts, 
                           param_combo, prefix = 'ts_')
    
    print(mtrc2.keys())
    
def test_gpc():
    param_grid = {
        'a': [1,2],
        'b': [3,4]
        }
    pc = pt.get_parameter_combos(param_grid)
    assert len(pc) == 4
    print(pc)
    for el in pc:
        assert isinstance(el, dict)

def test_holdout_res(X_tr, y_tr, X_ts, y_ts):
    
    X_train1, X_val1 = train_test_split(X_tr, test_size = 0.2, random_state = 0, stratify = y_tr)
    X_train2, X_val2 = train_test_split(X_tr, test_size = 0.2, random_state = 0, stratify = y_tr)
    assert X_train1.shape == X_train2.shape
    
    


def test_grid(X_tr, y_tr):
    param_grid = {               
        'max_depth': [3, 5, 7], 
        'min_samples_split': [2, 5]           
         }
    results = pt.grid(X_tr, y_tr, param_grid, model = RuleTreeClassifier)
    print(results)
    f1_macro = (results['val_f1_class_0'] + results['val_f1_class_1']) / 2
    assert f1_macro == results['val_f1']
    


if __name__ == '__main__':
    basepath = "C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/"
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    
    df_name, df = read_german_credit(basepath = basepath)
    X_tr, X_ts, y_tr, y_ts, feature_names, scaler = pt.split_data(df_name, df, save = False)
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
    
    baseline_param_grid = {
        'n_estimators': [100],
        'criterion': ['gini'],
        'max_depth': [20],
        'min_samples_split': [5], 
        'min_samples_leaf': [1]
        }
    
    parameter_combo = {              
        'max_depth': 5, 
        'min_samples_split': 2, 
        'min_samples_leaf': 1          
         }
    
    test_split_data(df_name, df, basepath, X_tr_df, X_tr, scaler, feature_names)
    test_get_metrics(X_tr, X_ts, y_tr, y_ts, metrics_to_compute, 
                     sens_feat_tr = X_tr_df['personal_status'].values, 
                     sens_feat_ts = X_ts_df['personal_status'].values
                     )
    
    test_gpc()
    test_holdout_res(X_tr, y_tr, X_ts, y_ts)
    test_grid(X_tr, y_tr)
    

















