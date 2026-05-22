# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 22:06:55 2026

@author: franc
"""


from sklearn.linear_model import LogisticRegression 
import pandas as pd




class lr(LogisticRegression):
    def __init__(self, features_lr=None, penalty='l2', dual=False, tol=1e-4, C=1.0, 
                 fit_intercept=True, intercept_scaling=1, class_weight=None, 
                 random_state=None, solver='lbfgs', max_iter=100, 
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
                 l1_ratio=None, **kwargs):
        
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                         class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter, multi_class=multi_class,
                         verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                         l1_ratio=l1_ratio, **kwargs)
        
        self.features_lr = features_lr
        
    def select_features(self, X):
        if self.features_lr is None:
            return X
        
        if isinstance(X, pd.DataFrame):
            if all(isinstance(feat, int) for feat in self.features_lr):
                return X.iloc[:, self.features_lr].values
            return X[self.features_lr].values
            
        return X[:, self.features_lr]
            
    def fit(self, X, y, sample_weight=None):
        return super().fit(self.select_features(X), y, sample_weight=sample_weight)
    
    def predict(self, X):
        return super().predict(self.select_features(X))
    
    def predict_proba(self, X):

        return super().predict_proba(self.select_features(X))

    