# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:31:17 2025

@author: franc
"""

import numpy as np
from joblib import load
import pytest
import Utils as ut


#predictions
def test_predictions(models, X_ts, y_true = None):
        
    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]
        assert isinstance(models, list)
        assert len(models) == 1
    
    if len(models) > 2:
        with pytest.raises(ValueError) as exc:
            ut.predictions(models, X_ts, limit_to_two = True)    
        assert "at most 2 models" in str(exc.value)
    
    
    if len(models) == 1:
        preds, probs = ut.predictions(models, X_ts)
        assert preds.shape == (X_ts.shape[0],) and probs.shape == (X_ts.shape[0], len(set(y_true))), 'Problem with predictions'


#apply distance metric
def test_adm():
    assert ut.apply_distance_metric(x = np.array([5,6,9]), y = np.array([5,6,9]), metric = ut.pearsonr) == 1
    assert ut.apply_distance_metric(x = np.array([5,6,9]), y = np.array([5,6,9]), metric = ut.mean_absolute_error) == 0
    assert ut.apply_distance_metric(x = np.array([5,6,9]), y = np.array([5,5,5]), metric = ut.pearsonr) == 0
    assert ut.apply_distance_metric(x = np.array([5,5,5]), y = np.array([5,5,5]), metric = ut.pearsonr) == 1
    
#prediction concordance filter
def dummy_pcf(preds1, preds2, to_filter, y_true = None, concordant = True):
    
    if y_true is None:
        
        condition = (preds1 == preds2) if concordant  else (preds1 != preds2) 
    else:
        condition = (preds1 == y_true) & (preds2 == y_true) if concordant else (preds1 != y_true) & (preds2 != y_true)
    
    if not isinstance(to_filter, (list, tuple)):
        return to_filter[condition]
    else:
        return [f[condition] for f in to_filter]

def test_pcf():
    np.random.seed(42)
    preds1 = np.array([1, 1, 0, 1, 0])
    preds2 = np.array([1, 0, 0, 1, 1])
    y_true = np.array([1, 1, 0, 1, 1])
    probs1 = np.array([1, 2, 3, 4, 5])
    probs2 = np.array([5, 4, 3, 2, 1])
    arr = np.random.rand(5, 5)
    
    conc_filt = [np.array([1, 3, 4]), np.array([5, 3, 2])]
    disc_filt = [np.array([2, 5]), np.array([4, 1])]
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = None, concordant = True)[0], conc_filt[0])
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = None, concordant = True)[1], conc_filt[1])
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = None, concordant = False)[0], disc_filt[0])
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = None, concordant = False)[1], disc_filt[1])
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = y_true)[0], conc_filt[0])
    np.testing.assert_array_equal(dummy_pcf(preds1, preds2, to_filter = [probs1, probs2], y_true = y_true)[1], conc_filt[1])
    assert isinstance(dummy_pcf(preds1, preds2, to_filter = arr, y_true = y_true), np.ndarray)





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
    models = [rtc_bin, lr_bin, knn_bin]
    
    #multiclass setting
    X_tr_mult = np.genfromtxt(f'{path}/vehicle_X_tr.csv', delimiter=',', skip_header=1)
    X_ts_mult = np.genfromtxt(f'{path}/vehicle_X_ts.csv', delimiter=',', skip_header=1)
    y_true_mult = np.genfromtxt(f'{path}/vehicle_y_ts.csv', delimiter=',', skip_header=1)
    
    rtc_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
    knn_mult = load(f'{path_models}/vehicle/vehicle_knn_1.joblib')
    lr_mult = load(f'{path_models}/vehicle/vehicle_lr_1.joblib')
    modelsm = [rtc_mult, rtc_mult]
    
    
    test_predictions(models = rtc_bin, X_ts = X_ts_bin, y_true = y_true_bin)
    test_predictions(models = models, X_ts = X_ts_bin)
    test_adm()
    test_pcf()
    
    
    
    