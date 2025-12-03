# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:13:28 2025

@author: franc
"""

import feature_importance as fi
import support_functions as sf
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau, spearmanr, pearsonr

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
modelsb = [m1_bin, m2_bin]
sensitive_features_bin = X_ts_bin_df['personal_status']


#multiclass setting
X_tr_mult = np.genfromtxt(f'{path}/vehicle_X_tr.csv', delimiter=',', skip_header=1)
X_ts_mult = np.genfromtxt(f'{path}/vehicle_X_ts.csv', delimiter=',', skip_header=1)
y_true_mult = np.genfromtxt(f'{path}/vehicle_y_ts.csv', delimiter=',', skip_header=1)

m1_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
m2_mult = load(f'{path_models}/vehicle/vehicle_rtc_1.joblib')
modelsm = [m1_mult, m2_mult]
X_ts_mult_df = pd.read_csv(f'{path}/vehicle_X_ts.csv')

sensitive_features_mult =pd.qcut(X_ts_mult_df['CIRCULARITY'], q=4, labels=[1, 2, 3, 4])


#importance array similarity
np.random.seed(42)
arr1_bin  = np.random.rand(5, 5)
arr2_bin = arr1_bin
arr1_mult = np.random.rand(5, 5, 4)
arr2_mult = arr1_mult

ias_bin = fi.importance_array_similarity(arr1_bin, arr2_bin, metric = pearsonr)
ias_mult = fi.importance_array_similarity(arr1_mult, arr2_mult, metric = mean_absolute_error)
ias_mult.shape
assert ias_mult.shape == (arr1_mult.shape[0], arr1_mult.shape[2]) and ias_bin.shape == (arr1_bin.shape[0],)
assert np.all(ias_bin) == 1
assert np.all(ias_mult) == 0

#get_shaps and shap similarity
shaps1_bin = fi.get_shaps(m1_bin, X_ts_bin, expl = fi.KernelExplainer)
shaps2_bin = fi.get_shaps(m1_bin, X_ts_bin, expl = fi.KernelExplainer)
shaps1_mult = fi.get_shaps(m1_mult, X_ts_mult, expl = fi.KernelExplainer)
shaps2_mult = fi.get_shaps(m1_mult, X_ts_mult, expl = fi.KernelExplainer)

assert shaps1_bin.all() == shaps2_bin.all()
assert shaps1_bin.ndim == 2
assert shaps1_mult.ndim == 3
assert shaps1_mult.shape[0] == X_ts_mult.shape[0] and shaps1_mult.shape[1] == X_ts_mult.shape[1] and shaps1_mult.shape[2] == len(set(y_true_mult))     

shapsim_bin = fi.shaps_similarity(modelsb, X_ts_bin)
shapsim_mult = fi.shaps_similarity(modelsm, X_ts_mult)
assert np.isnan(shapsim_bin).any() == False and np.isnan(shapsim_mult).any() == False



