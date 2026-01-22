# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:06:57 2025

@author: franc
"""


import pandas as pd
import numpy as np
from Utils import concatenate_df


#mi rende un'array con id dei modelli che rispettano la condizione
def get_rashomon(metrics_df, condition:str):
    filtered = metrics_df.query(condition)
    return np.array(filtered['model_id']), filtered

#è la frazione di modelli totali nel rashomon
def rashomon_ratio(metrics_df, condition:str):
    tot_models = metrics_df['model_id'].values
    rash_models, rash_df = get_rashomon(metrics_df, condition)

    rr = rash_models.shape[0]/tot_models.shape[0]
    return rr

def rr_variation(df, threshold, tol, metric):
    ratios = []
    for t in tol:
        cond = f'{metric} >= {threshold - (threshold*t)}'
        ratio = rashomon_ratio(df, cond)        
        ratios.append(ratio)
    return ratios

if __name__ == '__main__':
    full_df = concatenate_df(path = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\lr',
                      file_names = '*_holdout')

    cols_to_keep = [col for col in full_df.columns if col.startswith('val_') or col in ['n_neighbors']]
    df = full_df[cols_to_keep]
        
    condition = 'val_accuracy >= 0.7 and val_dem_parity <= 0.2'
    rashomon_ratio(full_df, condition)

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    