# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 20:53:13 2025

@author: franc
"""

import fairness as fns
from preprocess_and_training import split_data
from HybridReaders import read_german_credit, read_vehicle
import numpy as np
import pandas as pd
from joblib import load

basepath = "C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/"
path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
path_models = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models'


#binary setting
X_tr_bin = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
X_ts_bin = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
y_true_bin = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)

X_ts_bin_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')

m1_bin = load(f'{path_models}/german/german_credit_rtc_1.joblib')
m2_bin = load(f'{path_models}/german/german_credit_rtc_1.joblib')

sensitive_features_bin = X_ts_bin_df['personal_status']


#multiclass setting
X_tr_mult = np.genfromtxt(f'{path}/vehicle_X_tr.csv', delimiter=',', skip_header=1)
X_ts_mult = np.genfromtxt(f'{path}/vehicle_X_ts.csv', delimiter=',', skip_header=1)
y_true_mult = np.genfromtxt(f'{path}/vehicle_y_ts.csv', delimiter=',', skip_header=1)

m1_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
m2_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')

X_ts_mult_df = pd.read_csv(f'{path}/vehicle_X_ts.csv')

sensitive_features_mult =pd.qcut(X_ts_mult_df['CIRCULARITY'], q=4, labels=[1, 2, 3, 4])


#negative_predictive_value
def test_npv():
    y_pred1 = m1_bin.predict(X_ts_bin)
    y_pred2 = m2_bin.predict(X_ts_bin)
    
    npv1 = fns.negative_predictive_value(y_true = y_true_bin, y_pred = y_pred1) 
    npv2 = fns.negative_predictive_value(y_true = y_true_bin, y_pred = y_pred2)
    print(npv1)
    assert npv1 == npv2, 'Error: if the model is the same, the npv are supposed to be the same'
    
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 1] 
    val = fns.negative_predictive_value(y_true, y_pred)
    assert np.isclose(val, 2/3), f"Error: expected {2/3}, obtained {val}"

    y_true = [0, 1]
    y_pred = [1, 1] 
    assert fns.negative_predictive_value(y_true, y_pred) == 0.0, "Error: division by zero not managed"


#fairness_explorer: se binaria rende una serie pandas, se multiclass una lista di serie per ogni classe 
def test_fexp(fairness_metrics):
    
    expl_bin = fns.fairness_explorer(m1_bin, X_ts_bin, y_true_bin, sensitive_features_bin, fairness_metrics)
    expl_mult = fns.fairness_explorer(m1_mult, X_ts_mult, y_true_mult, sensitive_features_mult, fairness_metrics)
    print('BINARY EXPLAINER: \n', expl_bin)
    for i, el in enumerate(expl_mult):
        print(f'CLASS {i} EXPLAINER: \n', el)
    assert isinstance(expl_bin, pd.Series), 'Error: the result for binary classification should be a pd.Series'
    assert isinstance(expl_mult, list), 'Error: the result for multiclass classification should be a list'
    for el in expl_mult:
        assert isinstance(el, pd.Series), 'Error: every element for multiclass should be a pd.Series'
    
#advanced_fairness
def test_af():
    
    af_bin = fns.advanced_fairness(m1_bin, X_ts_bin, y_true_bin, sensitive_features_bin)
    af_mult = fns.advanced_fairness(m1_mult, X_ts_mult, y_true_mult, sensitive_features_mult)
    print('ADVANCED FAIRNESS FOR BINARY: \n', af_bin)
    print('ADVANCED FAIRNESS FOR MULTICLASS: \n', af_mult)
    for value in af_bin.values():
        assert isinstance(value, float)
    for value in af_mult.values():
        assert isinstance(value, list)
        assert len(value) == len(set(y_true_mult))






if __name__ == '__main__':
    basepath = "C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/"
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    path_models = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models'


    #binary setting
    X_tr_bin = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts_bin = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_true_bin = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)

    X_ts_bin_df = pd.read_csv(f'{path}/german_credit_X_ts.csv')

    m1_bin = load(f'{path_models}/german/german_credit_rtc_1.joblib')
    m2_bin = load(f'{path_models}/german/german_credit_rtc_1.joblib')

    sensitive_features_bin = X_ts_bin_df['personal_status']


    #multiclass setting
    X_tr_mult = np.genfromtxt(f'{path}/vehicle_X_tr.csv', delimiter=',', skip_header=1)
    X_ts_mult = np.genfromtxt(f'{path}/vehicle_X_ts.csv', delimiter=',', skip_header=1)
    y_true_mult = np.genfromtxt(f'{path}/vehicle_y_ts.csv', delimiter=',', skip_header=1)

    m1_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
    m2_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')

    X_ts_mult_df = pd.read_csv(f'{path}/vehicle_X_ts.csv')

    sensitive_features_mult =pd.qcut(X_ts_mult_df['CIRCULARITY'], q=4, labels=[1, 2, 3, 4])
    
    fairness_metrics = {
        'demographic parity(selection_rate)': fns.selection_rate,
        'equal opportunity(TPR)': fns.true_positive_rate,
        'predictive equality(FPR)': fns.false_positive_rate,
        'predictive parity(precision)': fns.precision_score,
        'negative predictive parity(NPV)': fns.negative_predictive_value}

    test_npv()
    test_fexp(fairness_metrics)
    test_af()




