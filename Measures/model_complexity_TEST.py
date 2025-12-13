# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:35:01 2025

@author: franc
"""

import model_complexity as mc
import numpy as np
from joblib import load
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression





#tree_depth
def test_td(root):
    depth = mc.tree_depth(root)
    assert depth == 2, f"Problem tree depth:\nExpected depth: 2, Obtained: {depth}"

#count nodes rtc    
def test_cn(root):
    count = mc.count_nodes_rtc(root)
    assert count == 5, f"Problem count nodes rtc:\nExpected node count: 5, Obtained: {count}"  

#path lengths
def test_path_lengths(model, X_ts):
        leaf_indexes = model.apply(X_ts).tolist() #leaf index per ogni instance
        assert len(leaf_indexes) == X_ts.shape[0]

        path_lengths = [len(idx) - 1 for idx in leaf_indexes]
        assert len(path_lengths) == X_ts.shape[0], 'Problem in path lengths'

#lr complexity
def test_lr_complexity(model, p = 2):
    coefficients = model.coef_
    n_classes = range(coefficients.shape[0])
    print(n_classes)
    if len(n_classes) == 1:
        magnitude = np.linalg.norm(coefficients, ord = p)
        assert isinstance(magnitude, float)
    
    #calcolo le misure classe per classe
    else:
        magnitude = [np.linalg.norm(coefficients[cl], ord = p) for cl in n_classes]
        assert isinstance(magnitude, list)
        assert len(magnitude) == len(n_classes)



if __name__ == '__main__':
    
    path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
    path_models = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models'
    rtc = load(f'{path_models}/german/german_credit_rtc_2.joblib')
    knn = load(f'{path_models}/german/german_credit_knn_1.joblib')
    X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)

    leaf1 = {"is_leaf": True, "left_node": None, "right_node": None}
    leaf2 = {"is_leaf": True, "left_node": None, "right_node": None}
    leaf3 = {"is_leaf": True, "left_node": None, "right_node": None}

    nodeA = {"is_leaf": False, "left_node": leaf1, "right_node": leaf3}
    root = {"is_leaf": False, "left_node": nodeA, "right_node": leaf2}
    
    X_bin, y_bin = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=0)
    X_mult, y_mult = make_classification(
                                        n_samples=300,
                                        n_features=10,
                                        n_informative=8,
                                        n_redundant=2,
                                        n_repeated=0,
                                        n_classes=3,
                                        n_clusters_per_class=2,
                                        random_state=0
                                        )
    
    lr_bin = LogisticRegression().fit(X_bin, y_bin)
    lr_mult = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_mult, y_mult)
    
    print('KNN local complexity: \n', mc.knn_local_complexity(knn))
    print('KNN model: \n', knn)
    print('binary lr complexity: \n', test_lr_complexity(lr_bin, multiclass=False))
    print('multiclass lr complexity: \n', test_lr_complexity(lr_mult, multiclass=True))
    test_td(root)
    test_cn(root)
    test_path_lengths(rtc, X_ts)
    test_lr_complexity(lr_bin)
    test_lr_complexity(lr_mult)








