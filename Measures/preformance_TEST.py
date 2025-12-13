# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 23:56:46 2025

@author: franc
"""

from joblib import load
import numpy as np
import performance as pf
from sklearn.metrics import accuracy_score, recall_score, jaccard_score
import inspect
import pytest


def test_pf(model, X_ts, y_true):

    assert isinstance(pf.performance(model, X_ts, y_true, measure = accuracy_score, multiclass = True), float)
    assert pf.performance(model, X_ts, y_true, measure = recall_score, multiclass = True, average = None).shape[0] == len(set(y_true))
    assert isinstance(pf.performance(model, X_ts, y_true, measure = recall_score, multiclass = True, average = 'weighted'), float)
    
def test_pd(models:list, X_ts, y_true, measure = accuracy_score, average = None):
        
        scores = [pf.performance(model, X_ts, y_true, measure = measure, average = average) for model in models]
        
        
        if len(scores) > 2:
            with pytest.raises(ValueError) as exc:
                pf.performance_difference(models, X_ts, y_true)
            assert 'at most two models' in str(exc.value)
        
def test_ps(models, X_ts, y_true):
    if len(np.unique(y_true)) > 2:
        assert pf.predictions_similarity(models, X_ts, y_true, average = None).shape[0] == len(set(y_true))
        assert isinstance(pf.predictions_similarity(models, X_ts, y_true, average = 'weighted'), float)
    else:
        exit

def test_cs(models, X_ts, y_true):
    if len(set(y_true)) > 2:
        assert isinstance(pf.confidence_similarity(models, X_ts, y_true) , np.ndarray)
        assert pf.confidence_similarity(models, X_ts, y_true).shape[0] == len(set(y_true))
    else:
        assert isinstance(pf.confidence_similarity(models, X_ts, y_true) , float)
        
def test_tcp(models, X_ts, y_true, concordant = False):

    assert pf.true_class_probability(models, X_ts, y_true, concordant = concordant).shape[1] == 2



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
    
    test_pf(rtc_mult, X_ts_mult, y_true_mult)
    test_pd(models = [rtc_bin, lr_bin, knn_bin], X_ts = X_ts_bin, y_true = y_true_bin)
    test_ps(modelsm, X_ts_mult, y_true_mult)
    test_cs(modelsm, X_ts_mult, y_true_mult)
    test_cs(modelsb, X_ts_bin, y_true_bin)
    test_tcp(modelsb, X_ts_bin, y_true_bin)

    
    
    
    
    
    
    
    
    
    
    