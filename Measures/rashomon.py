# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:06:57 2025

@author: franc
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fairness as fn
import performance as pf
import robustness as rb
from joblib import load


#mi rende un'array con id dei modelli che rispettano la condizione
def get_rashomon(metrics_df, condition:str):
    filtered = metrics_df.query(condition)
    return np.array(filtered['model_id'])



def plot_rashomon(metrics_df, x_var, y_var, z_var, rashomon_models):
    fig = plt.figure(figsize=(10, 8), layout = 'constrained')
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        metrics_df[x_var],
        metrics_df[y_var],
        metrics_df[z_var],
        c=metrics_df['model_id'].isin(rashomon_models).map({True: 'blue', False: 'gray'}),
        s=50)
    
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    ax.set_title("Rashomon Set")
    ax.view_init(elev=20, azim=-10) 
    plt.show()
    



if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\rtc_mean_results.csv')
    print(df.columns)
    min(df['val_acc_robustness'].values)
    
    condition = 'val_accuracy >= 0.7 and val_dem_parity <= 0.2'
    rashomon = get_rashomon(df, condition)
    rashomon
    
    
    x_var = 'val_accuracy'
    y_var = 'val_dem_parity'
    z_var = 'val_acc_robustness'
    plot_rashomon(df, x_var, y_var, z_var, rashomon)
    
    m1 = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\german_rtc_1.joblib')
    m1 = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\german_rtc_5.joblib')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    