# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 19:27:24 2025

@author: franc
"""


from knn import knn, features_combos
import numpy as np


def test_knn():
    m = knn(features = [2, 5, 6])
    m.fit(X_tr, y_tr)
    assert knn().__class__.__name__ == 'knn'
    assert m.n_features_in_ == len(m.features)
    preds = m.predict(X_ts)
    probs = m.predict_proba(X_ts)
    assert preds.shape == (X_ts.shape[0],) if len(np.unique(y_ts)) == 2 else (X_ts.shape[0], len(np.unique(y_ts)))
    assert probs.shape == (X_ts.shape[0], len(np.unique(y_ts)))
    
def test_feat_combos():
    p = [1, 2]
    n_neighbors = [5, 10, 2, 13, 7]
    weights = ['uniform', 'distance']
    n_total_combos = 1000
    f = features_combos(X_tr, p = p, n_neighbors = n_neighbors, weights = weights, n_total_combos = n_total_combos)
    
    for el in f:
        assert len(el) >= 2
    assert len(f) == len(set(tuple(x) for x in f))
    
    

    

if __name__ == '__main__':
    
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    X_tr = np.genfromtxt(f'{path}/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_ts = np.genfromtxt(f'{path}/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    y_tr = np.genfromtxt(f'{path}/german_credit_y_tr.csv', delimiter=',', skip_header=1)
    
    test_knn()
    test_feat_combos()
    
    p = [1, 2]
    n_neighbors = [3, 6, 9, 12, 16]
    weights = ['uniform', 'distance']
    n_total_combos = 1000
    n_base_combos = len(p)*len(weights)*len(n_neighbors)
    features_combos(X_tr, p = p, n_neighbors = n_neighbors, weights = weights, n_total_combos = n_total_combos, prt = True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    