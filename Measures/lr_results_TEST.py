# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 10:13:29 2026

@author: franc
"""


from joblib import load
import lr_results as res



def test_add_noise():
    model = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\lr\00d7b0a83a0da47ad480428c6c5f1160.joblib')
    seed = 0 
    std = 1 
    print(model.coef_)
    res.add_noise_lr(model, seed, std)
    print(model.coef_)
    
    
if __name__ == '__main__':
    test_add_noise()