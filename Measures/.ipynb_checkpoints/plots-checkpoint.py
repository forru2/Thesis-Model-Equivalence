# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 15:09:33 2025

@author: franc
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
from scipy.stats import kendalltau, spearmanr, pearsonr
from Utils import apply_distance_metric
import pacmap
import pandas as pd

#fa il plot delle matrici di correlazione
def plot_corr_matrix(corr_matrix, figsize:tuple = (12, 10), annot = True, title = None, save_plot = False, file_dir = '.'):
    plt.figure(figsize = figsize)
    sns.heatmap(corr_matrix, annot = annot, fmt = '.2f', cmap = 'coolwarm')
    if title:
        plt.title(title)
    if save_plot:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok = True)
        plt.savefig(os.path.join(file_dir, f'{title}.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    


#fa una griglia di corr matrices
def plot_corr_matrix_grid(corr_matrices:list, titles:list, cols=2, figsize_unit=(7, 6), annot = False):

    n_matrices = len(corr_matrices)
    rows = math.ceil(n_matrices / cols)
    
    fig_width = cols * figsize_unit[0]
    fig_height = rows * figsize_unit[1]
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for i in range(n_matrices):
        ax = axes[i]
        corr = corr_matrices[i]
        
        sns.heatmap(corr, annot = annot, fmt = '.2f', cmap ='coolwarm', ax = ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


    
    
#plot di prima visualizzazione del rashomon set
def plot_rashomon(df, x, y, z, rash_models, title = 'Rashomon Set'):
    fig = plt.figure(figsize=(10, 8), layout = 'constrained')
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        df[x],
        df[y],
        df[z],
        c=df['model_id'].isin(rash_models).map({True: 'blue', False: 'gray'}),
        s=50)
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)
    ax.view_init(elev=20, azim=-10) 
    plt.show()
    
    
    
#plot degli scatter plot per vedere la variazione di correlazione    
def plot_corr_scatter(rashomon_df, not_rashomon_df, x:str, y:str, titles:list, corr_metric = pearsonr, figsize = (8, 6)):
    
    corr_rash = apply_distance_metric(rashomon_df[x], rashomon_df[y], metric = corr_metric)
    corr_not_rash = apply_distance_metric(not_rashomon_df[x], not_rashomon_df[y], metric = corr_metric)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)
    
    ax1.scatter(not_rashomon_df[x], not_rashomon_df[y], alpha = 0.3,
                color = 'grey', label = f'Corr. outside Rashomon: {round(corr_not_rash, 2)}')

    ax1.scatter(rashomon_df[x], rashomon_df[y], alpha = 0.8,
                color = 'blue', label = f'Corr. inside Rashomon: {round(corr_rash, 2)}')

    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax1.set_title(titles[0])
    ax1.legend()
    
    ax2.scatter(rashomon_df[x], rashomon_df[y], alpha = 0.8, color = 'blue')
    
    x_min, x_max = rashomon_df[x].min(), rashomon_df[x].max()
    y_min, y_max = rashomon_df[y].min(), rashomon_df[y].max()
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    ax2.set_xlim(x_min - margin_x, x_max + margin_x)
    ax2.set_ylim(y_min - margin_y, y_max + margin_y)
    
    ax2.set_xlabel(x)
    ax2.set_ylabel(y)
    ax2.set_title(titles[1])
    plt.tight_layout()
    plt.show()    
    
    

#plot del reachability graph per OPTICS
def plot_reachability_graph(clustering):
    space = np.arange(len(clustering.labels_))
    reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_[clustering.ordering_]

    plt.figure(figsize=(12, 7))
    
    raw_palette = sns.color_palette("tab20", len(np.unique(labels)))
    color_map = {label: raw_palette[i] for i, label in enumerate(np.unique(labels))}
    if -1 in color_map:
        color_map[-1] = (0.2, 0.2, 0.2, 0.5)
        
    
    for klass in color_map.keys():
        cluster_name = f'Cluster {klass}' if klass != -1 else 'Noise'
        x_values = space[labels == klass]
        y_values = reachability[labels == klass]
        plt.plot(x_values, y_values, color=color_map[klass], marker='.', linestyle='None', alpha=0.7, label = cluster_name)
        

    plt.ylabel('Reachability Distance')
    plt.title('Reachability Graph')
    plt.legend(loc = 'upper right', bbox_to_anchor = (1.15, 1))
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.show()




def plot_clustering(df, clustering, title = ''):
    
    labels = clustering.labels_[clustering.ordering_]
    plt.figure(figsize=(12, 8))

    raw_palette = sns.color_palette('tab20', len(np.unique(labels)))
    color_map = {label: raw_palette[i] for i, label in enumerate(np.unique(labels))}
    if -1 in color_map:
        color_map[-1] = (0.2, 0.2, 0.2, 0.5)
        
    for klass in color_map.keys():
        mask = (labels == klass)
        plt.scatter(
            df.iloc[:, 0][mask], 
            df.iloc[:, 1][mask],
            color = color_map[klass],
            label=f'Cluster {klass}' if klass != -1 else 'Noise',
            s=40,
            alpha=0.8,
            linewidth=0.5
        )
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()





    