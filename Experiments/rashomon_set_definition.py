# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:06:57 2025

@author: franc
"""


import pandas as pd
import numpy as np
from Utils import concatenate_df
import matplotlib.pyplot as plt
import os


#mi rende un'array con id dei modelli che rispettano la condizione e il df filtrato
def get_rashomon(metrics_df, condition:str):
    filtered = metrics_df.query(condition)
    return np.array(filtered['model_id']), filtered

#calcola il Rashomon Ratio
def rashomon_ratio(metrics_df, condition:str):
    tot_models = metrics_df['model_id'].values
    rash_models, _ = get_rashomon(metrics_df, condition)

    rr = rash_models.shape[0]/tot_models.shape[0]
    return rr

#rende una lista di rashomon ratios al variare del livello di tolleranza
def rr_variation(df, base_threshold, tol:list, metric):
    ratios = []
    for t in tol:
        cond = f'{metric} >= {base_threshold - (base_threshold*t)}'
        ratio = rashomon_ratio(df, cond)        
        ratios.append(ratio)
    return ratios

def plot_rr_variation(data, title = 'Variation of the RR wrt Tolerance on F1', save = False, file_path = '.', valsize = 10, labelsize = 10,
                     legendsize = 15, titlesize = 15):
    for el in data:
        plt.plot(np.linspace(0, 1, 50), el['rr_var'], label = el['fam'])
    plt.grid()
    plt.xticks(fontsize = valsize)
    plt.yticks(fontsize = valsize)
    plt.legend(fontsize = legendsize)
    plt.xlabel('Tolerance', fontsize = labelsize)
    plt.ylabel('Rashomon Ratio', fontsize = labelsize)
    plt.title(title, fontsize = titlesize, fontweight = 'bold')

    if save:
        plt.savefig(os.path.join(file_path, f'{title}.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_rashomon_curve(all_data, epsilons=[0.01, 0.05, 0.10], n_rtc=1000, n_knn = 1000, n_lr=1040, print_xlab = True,
                             save = False, file_path = '.', filename = ''):

    figsize=(24, 6)
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    markers = ['*', '^', 'o']  
    colors = ['#984ea3', '#377eb8', '#e41a1c'] 
    eps_to_plot = sorted(epsilons, reverse=True) 

    for i, ax in enumerate(axs):
        model_data = all_data[i]
        curve = model_data['metric']
        if model_data['fam'] == 'LR':
            n_total = n_lr
        elif model_data['fam'] == 'DT':
            n_total = n_rtc
        else:
            n_total = n_knn
        
        ax.plot(curve, color='green', linewidth=3)        
        for idx, eps in enumerate(eps_to_plot):
            rs_models = model_data['rs_models'][eps]
            start_index = n_total-(rs_models)
            start_index = max(0, min(start_index, len(curve) - 1))
            y_val = curve[start_index]
            
            ax.axvline(x=start_index, color=colors[idx], linestyle='--', linewidth=1.0)
            ax.scatter(start_index, y_val, color=colors[idx], marker=markers[idx], 
                        s=200, zorder=5, edgecolors='black', label=f'tol = {eps}')

            y_stack_offset = 15 + (idx * 25)             
            ax.annotate(f'{rs_models} models', 
                        xy=(start_index, y_val), 
                        xytext=(0, y_stack_offset), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=23,
                        fontweight='bold', 
                        color=colors[idx],
                        arrowprops=dict(arrowstyle='-', color=colors[idx], lw=1, alpha=0.5),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

        ax.set_title(f"{model_data['fam']}", fontsize=25, fontweight='bold', pad=10)
        if print_xlab:
            ax.set_xlabel('Number of Models', fontsize=25)
            ax.tick_params(axis='both', labelsize=20)
        else:
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(labelbottom=False)
        if i == 0: ax.set_ylabel('F1 Score', fontsize=25)
        ax.grid(True, linestyle=':')
        ax.set_ylim(-0.05, 1.0)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.1), 
               ncol=3,                     
               fontsize=30, 
               frameon=False)

    if save:
        plt.savefig(os.path.join(file_path, f'{filename}.png'), dpi = 300, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    full_df = concatenate_df(path = r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\german\lr',
                      file_names = '*_holdout')

    cols_to_keep = [col for col in full_df.columns if col.startswith('val_') or col in ['n_neighbors']]
    df = full_df[cols_to_keep]
        
    condition = 'val_accuracy >= 0.7 and val_dem_parity <= 0.2'
    rashomon_ratio(full_df, condition)

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    