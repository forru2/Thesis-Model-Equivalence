# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:13:28 2025

@author: franc
"""

import feature_importance as fi
import Utils as sf
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau, spearmanr, pearsonr


#importance array similarity
def test_ias():
    np.random.seed(42)
    arr1_bin  = np.random.rand(5, 5)
    arr2_bin = arr1_bin
    arr1_mult = np.random.rand(5, 5, 4)
    arr2_mult = arr1_mult
    ias_bin = fi.importance_array_similarity(arr1_bin, arr2_bin, metric = pearsonr)
    ias_mult = fi.importance_array_similarity(arr1_mult, arr2_mult, metric = mean_absolute_error)
    assert ias_mult.shape == (arr1_mult.shape[0], arr1_mult.shape[2]) and ias_bin.shape == (arr1_bin.shape[0],), 'Problem importance_array_similarity'
    assert np.all(ias_bin) == 1, 'Problem importance_array_similarity'
    assert np.all(ias_mult) == 0, 'Problem importance_array_similarity'
  
#feature importance rtc
def test_firtc(model, X_ts, y_true):
    arr = fi.feature_importance_rtc(model, X_ts, y_true)
    if len(np.unique(y_true)) > 2:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1], len(set(y_true))), 'Problem feature importance rtc'
    else:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1]), 'Problem feature importance rtc'
       
        
#feature importance lr
def test_filr(model, X_ts, y_true):
    arr = fi.feature_importance_lr(model, X_ts, y_true)
    if len(np.unique(y_true)) > 2:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1], len(set(y_true))), 'Problem feature importance lr'
    else:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1]), 'Problem feature importance lr'
    
    
#get_shaps and shap similarity
def test_gs(model, X_ts, y_true):
    
    arr = fi.get_shaps(model, X_ts, expl = fi.KernelExplainer)
    if len(np.unique(y_true)) > 2:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1], len(set(y_true))), 'Problem get shaps'
    else:
        assert arr.shape == (X_ts.shape[0], X_ts.shape[1]), 'Problem get shaps'
    
def test_ss(models, X_ts):
        
    shapsim = fi.shaps_similarity(models, X_ts)
    assert np.isnan(shapsim).any() == False
    assert shapsim.all() == 1



if __name__ == '__main__':
    basepath = "C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/"
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    path_models = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models'


    #binary setting
    X_tr_bin = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts_bin = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_true_bin = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)


    rtc_bin = load(f'{path_models}/german/german_credit_rtc_2.joblib')
    knn_bin = load(f'{path_models}/german/german_credit_knn_1.joblib')
    lr_bin = load(f'{path_models}/german/german_credit_lr_1.joblib')
    modelsb = [rtc_bin, rtc_bin]


    #multiclass setting
    X_tr_mult = np.genfromtxt(f'{path}/vehicle_X_tr.csv', delimiter=',', skip_header=1)
    X_ts_mult = np.genfromtxt(f'{path}/vehicle_X_ts.csv', delimiter=',', skip_header=1)
    y_true_mult = np.genfromtxt(f'{path}/vehicle_y_ts.csv', delimiter=',', skip_header=1)

    rtc_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
    knn_mult = load(f'{path_models}/vehicle/vehicle_knn_1.joblib')
    lr_mult = load(f'{path_models}/vehicle/vehicle_lr_1.joblib')
    modelsm = [rtc_mult, rtc_mult]
    
    test_ias()
    test_firtc(rtc_bin, X_ts_bin, y_true_bin)
    test_firtc(rtc_mult, X_ts_mult, y_true_mult)
    test_gs(rtc_bin, X_ts_bin, y_true_bin)
    test_gs(rtc_mult, X_ts_mult, y_true_mult)
    test_ss(modelsb, X_ts_bin)
    test_ss(modelsm, X_ts_mult)
    
    
    
    
    
    
    
    
    
    
    
    
    

