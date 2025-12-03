# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:35:01 2025

@author: franc
"""

import model_complexity as mc
import numpy as np

#modelli finti
class MockKNN:
    def __init__(self, k):
        self.n_neighbors = k
        
class MockTreeModel:
    def apply(self, X):
        return np.array([
            [0, 1, 4], 
            [0, 2]
        ], dtype=object)

    def get_leaf_nodes(self):
        return [4, 2, 5]
        
class MockLR:
    def __init__(self, coef):
        self.coef_ = np.array(coef)
        
#KNN
k = 5
knn = MockKNN(k)
res = mc.knn_local_complexity(knn)
assert res == k


#Tree
leaf1 = {"is_leaf": True, "left_node": None, "right_node": None}
leaf2 = {"is_leaf": True, "left_node": None, "right_node": None}
leaf3 = {"is_leaf": True, "left_node": None, "right_node": None}

nodeA = {"is_leaf": False, "left_node": leaf1, "right_node": leaf3}
root = {"is_leaf": False, "left_node": nodeA, "right_node": leaf2}

depth = mc.tree_depth(root)
assert depth == 2, f"Expected depth: 2, Obtained: {depth}"
 
count = mc.count_nodes_rtc(root)
assert count == 5, f"Expected node count: 5, Obtained: {count}"  

rtc = MockTreeModel()
X_dummy = np.zeros((2, 5)) 
complexity = mc.tree_complexity(rtc, root, X_dummy)

assert complexity['depth'] == 2
assert complexity['number_of_nodes'] == 5
assert complexity['number_of_rules'] == 3
np.testing.assert_array_equal(complexity['path_lengths'], np.array([2, 1]))


#log reg      
coef = [2, 0, -2]
lrb = MockLR(coef)
res = mc.lr_complexity(lrb, p=2, multiclass=False)

assert res['nonzero_coefficients'] == 2
assert res['magnitude'] == np.linalg.norm(coef)

coef = [[1, 0, 0],[0, 3, 4]]
lrm = MockLR(coef)
res = mc.lr_complexity(lrm, p=2, multiclass=True)

assert res['nonzero_coefficients'] == [1, 2]
assert res['magnitude'] == [1.0, 5.0]













