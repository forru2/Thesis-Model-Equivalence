# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 00:16:36 2025

@author: franc
"""

import numpy as np



def general_tree_depth(node, **kwargs):
    if node is None or node["is_leaf"]:
        return 0
    
    left_depth = general_tree_depth(node["left_node"])
    right_depth = general_tree_depth(node["right_node"])
    
    return 1 + max(left_depth, right_depth)

#calcola l'altezza del rtc
def rtc_depth(model, **kwargs):
    node = model.get_rules()
    return general_tree_depth(node)



def general_count_nodes(node, **kwargs):
    if node is None:
        return 0
    
    left_count = general_count_nodes(node["left_node"])
    right_count = general_count_nodes(node["right_node"])
    
    return 1 + left_count + right_count

#conta il numero di nodi
def count_nodes_rtc(model, **kwargs):
    node = model.get_rules()
    return general_count_nodes(node)

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
        'depth': rtc_depth(node),
        'number_of_nodes': count_nodes_rtc(node),
        'number_of_rules': n_rules(model),
        'path_lengths': path_lengths(model, X_ts)
        }


#valuta la complessità del lr considerando la magnitudine dei coeff
def lr_complexity(model, p = 2, average = True, **kwargs):
    
    coefficients = model.coef_
    magnitude = np.linalg.norm(coefficients, ord = p, axis = 1)
    
    if magnitude.shape[0] == 1:
        return magnitude[0]
    
    if average:
        return np.mean(magnitude)
    
    return magnitude
    







