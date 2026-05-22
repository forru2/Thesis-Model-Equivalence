# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 11:32:16 2026

@author: franc
"""


from sklearn.neighbors import KNeighborsClassifier
import pandas as pd




class knn(KNeighborsClassifier):
    def __init__(self, features_knn = None, n_neighbors = 5, weights = 'uniform', 
                 algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski', 
                 metric_params = None, n_jobs = None, **kwargs):
        
        super().__init__(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm,
                         leaf_size = leaf_size, p = p, metric = metric,
                         metric_params = metric_params, n_jobs = n_jobs, **kwargs)
        self.features_knn = features_knn
        
    def select_features(self, X):
        if self.features_knn is None or X.shape[1] == len(self.features_knn):
            return X
        
        if isinstance(X, pd.DataFrame):
            if all(isinstance(feat, int) for feat in self.features_knn):
                return X.iloc[:, self.features_knn].values
            return X[self.features_knn].values
            
        return X[:, self.features_knn]
            
    
    def fit(self, X, y):
        return super().fit(self.select_features(X), y)
    
    def predict(self, X):
        return super().predict(self.select_features(X))
    
    def predict_proba(self, X):
        return super().predict_proba(self.select_features(X))
    
    