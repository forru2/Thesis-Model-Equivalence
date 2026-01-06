# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 23:31:56 2026

@author: franc
"""


import rtc_results as res
import numpy as np
from RuleTree import RuleTreeClassifier

def test_rbs():
    
    parameter_combo = {              
        'max_depth': 5, 
        'min_samples_split': 2, 
        'min_samples_leaf': 1          
         }
    random_state = 0
    rbs = res.manage_rtc_base_stumps(parameter_combo, random_state)
    assert 'stump_selection' in rbs and 'base_stumps' in rbs
    
def test_rtc_node_diversity(parameter_combo, X_tr, y_tr, X_ts):
    rng_test = np.random.RandomState(0)
    param_combo_test = res.manage_rtc_base_stumps(parameter_combo, rng_test)
    
    
    m1 = RuleTreeClassifier(**param_combo_test, random_state=0)
    m1.fit(X_tr, y_tr)
    
    m2 = RuleTreeClassifier(**param_combo_test, random_state=0)
    m2.fit(X_tr, y_tr)
    
    assert np.sum(m1.predict(X_ts) != m2.predict(X_ts)), 'the code is not working'
    
    

    
if __name__ == '__main__':
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    
    parameter_combo = {              
        'max_depth': 5, 
        'min_samples_split': 2, 
        'min_samples_leaf': 1          
         }
    test_rbs()
    test_rtc_node_diversity(parameter_combo, X_tr, y_tr, X_ts)
    