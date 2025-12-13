# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:16:36 2025

@author: franc
"""

import numpy as np

#mi rende il numero di vicini
def knn_local_complexity(model, **kwargs):
    return model.n_neighbors

#calcola l'altezza del rtc
def tree_depth(node, **kwargs):
    if node is None or node["is_leaf"]:
        return 0
    
    left_depth = tree_depth(node["left_node"])
    right_depth = tree_depth(node["right_node"])
    
    return 1 + max(left_depth, right_depth)


#conta il numero di nodi
def count_nodes_rtc(node, **kwargs):
    if node is None:
        return 0
    
    left_count = count_nodes_rtc(node["left_node"])
    right_count = count_nodes_rtc(node["right_node"])
    
    return 1 + left_count + right_count

#calcola la lunghezza dei path per ogni instance
def path_lengths(model, X_ts, **kwargs):
        leaf_indexes = model.apply(X_ts).tolist() #leaf index per ogni instance (ex. Rrrl)
        path_lengths = [len(idx) - 1 for idx in leaf_indexes]     
        return np.array(path_lengths)
    
def n_rules(model, **kwargs):
    return len(model.get_leaf_nodes())

#valuta la complessità del rtc consideranco la profondità, il numero di nodi e il numero di regole
def tree_complexity(model, node, X_ts, **kwargs):
    return {
        'depth': tree_depth(node),
        'number_of_nodes': count_nodes_rtc(node),
        'number_of_rules': n_rules(model),
        'path_lengths': path_lengths(model, X_ts)
        }


#valuta la complessità del lr considerando il numero di nonzero coeff e la magnitudine dei coeff
def lr_complexity(model, p = 2, **kwargs):
    coefficients = model.coef_
    n_classes = range(coefficients.shape[0])
    
    if len(n_classes) == 1:
        magnitude = np.linalg.norm(coefficients, ord = p)
    
    #calcolo le misure classe per classe
    else:
        #nonzero_coefficients = [np.count_nonzero(coefficients[cl]) for cl in n_classes]
        magnitude = [np.linalg.norm(coefficients[cl], ord = p) for cl in n_classes]
        
    return magnitude