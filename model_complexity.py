# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:16:36 2025

@author: franc
"""

import numpy as np

#mi rende il numero di vicini
def knn_local_complexity(model):
    return model.n_neighbors

#calcola l'altezza del rtc
def tree_depth(node):
    if node is None or node["is_leaf"]:
        return 0
    
    left_depth = tree_depth(node["left_node"])
    right_depth = tree_depth(node["right_node"])
    
    return 1 + max(left_depth, right_depth)


#conta il numero di nodi
def count_nodes_rtc(node):
    if node is None:
        return 0
    
    left_count = count_nodes_rtc(node["left_node"])
    right_count = count_nodes_rtc(node["right_node"])
    
    return 1 + left_count + right_count

#calcola la lunghezza dei path per ogni instance
def path_lengths(model, X_ts):
        leaf_indexes = model.apply(X_ts)
        n_instances = range(leaf_indexes.shape[0])
        path_lengths = [len(leaf_indexes[idx]) - 1 for idx in n_instances]
        
        return np.array(path_lengths)

#valuta la complessità del rtc consideranco la profondità, il numero di nodi e il numero di regole
def tree_complexity(model, node, X_ts):
    return {
        'depth': tree_depth(node),
        'number_of_nodes': count_nodes_rtc(node),
        'number_of_rules': len(model.get_leaf_nodes()),
        'path_lengths': path_lengths(model, X_ts)
        }


#valuta la complessità del lr considerando il numero di nonzero coeff e la magnitudine dei coeff
def lr_complexity(model, p = 2, multiclass = False):
    coefficients = model.coef_
    
    if not multiclass:
        nonzero_coefficients = np.count_nonzero(coefficients)
        magnitude = np.linalg.norm(coefficients, ord = p)
    
    #calcolo le misure classe per classe
    else:
        n_classes = range(coefficients.shape[0])
        nonzero_coefficients = [np.count_nonzero(coefficients[cl]) for cl in n_classes]
        magnitude = [np.linalg.norm(coefficients[cl], ord = p) for cl in n_classes]
        
    return {
            'nonzero_coefficients': nonzero_coefficients,
            'magnitude': magnitude
            }
