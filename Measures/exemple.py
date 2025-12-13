# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:57:21 2025

@author: franc
"""

import pandas as pd
from joblib import load
import numpy as np
import rashomon as rm
import feature_importance as fi

path = 'C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati'
df = pd.read_csv(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Results\rtc_mean_results.csv')
#print(df.columns)
#min(df['val_acc_robustness'].values)
condition = 'val_accuracy >= 0.7 and val_dem_parity <= 0.2'
x_var = 'val_accuracy'
y_var = 'val_dem_parity'
z_var = 'val_acc_robustness'
m1 = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\german_rtc_1.joblib')
m2 = load(r'C:\Users\franc\OneDrive\Magistrale\Thesis-Model-Equivalence\Models\german\german_rtc_5.joblib')
X_ts = np.genfromtxt(f'{path}/german_credit_X_ts.csv', delimiter=',', skip_header=1)

rashomon = rm.get_rashomon(df, condition)
rashomon
rm.plot_rashomon(df, x_var, y_var, z_var, rashomon)

shap_sim = fi.shaps_similarity([m1, m2], X_ts)
print("Quando i due modelli concordano nemme previsioni, quanto si somigliano per l'importanza che danno alle feature?")
print('SHAPS SIMILARITY: \n', shap_sim)
print('E nel caso della instance 3?')
print(shap_sim[2])