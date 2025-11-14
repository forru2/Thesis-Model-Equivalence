# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 13:11:57 2025

@author: franc
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from RuleTree import RuleTreeClassifier
from HybridReaders import read_wdbc, read_compass, read_german_credit, read_vehicle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, recall_score, precision_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr, pearsonr
import random as rd
from joblib import dump, load
from shap import TreeExplainer, LinearExplainer, KernelExplainer
#import os
import inspect
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equal_opportunity_difference, equal_opportunity_ratio, equalized_odds_difference, equalized_odds_ratio
from fairlearn.metrics import selection_rate, MetricFrame
#from fairlearn import metrics
#from aif360.metrics import ClassificationMetric
#from aif360.datasets import BinaryLabelDataset

#split df and save
def split_ts(df_name, df, save = True, scale = True, test_size = 0.3, save_path = './'):
    
    y = df['y'].values
    X = df[df.columns[:-1]].values
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = test_size, random_state = 42, stratify = y)
    to_save = {f'{df_name}_X_tr': X_tr, f'{df_name}_X_ts': X_ts, f'{df_name}_y_tr': y_tr, f'{df_name}_y_ts': y_ts}
    
    if save:
        for key, el in to_save.items():
            try:
                pd.DataFrame(el).to_csv(f'{save_path}/{key}.csv', index=False)
            except FileNotFoundError:
                print('The directory does not exist or it is wrong')
        return X_tr, X_ts, y_tr, y_ts
    else:
        return X_tr, X_ts, y_tr, y_ts



#model training and saving
def train_models(model_type, df_name, X_tr, X_ts, y_tr, y_ts, max_models = 5, min_d = 3, max_d = 20, save_path = './', max_nb = 5, min_nb = 1):

    np.random.seed(0)
    n_neighbors = np.random.choice(np.arange(min_nb, max_nb +1), size = max_models, replace = False)
    C = np.random.choice(np.logspace(-4, 4), size = max_models, replace = False)
    for i in range(max_models):
        rd.seed(i)
        if model_type == 'rtc':
            model = RuleTreeClassifier(
                max_depth = rd.randint(min_d, max_d),
                criterion = rd.choice(('gini', 'entropy')),
                prune_useless_leaves=True
            )
            params = ['max_depth', 'criterion']
            
        elif model_type == 'knn':
            model = knn(
                n_neighbors = n_neighbors[i],
                weights = rd.choice(('uniform','distance'))
            )
            params = ['n_neighbors', 'weights']

        elif model_type == 'lr':
            model = lr(
                C = C[i],
                max_iter = 500
            )
            params = ['C']

        model.fit(X_tr, y_tr)
        try:
            dump(model, f'{save_path}/{df_name}_{model_type}_{i+1}.joblib')
        
        except FileNotFoundError:
            print('The directory does not exist or it is wrong')
            break
 
        #solo per dare una prima occhiata veloce
        y_pred = model.predict(X_ts)
        print(f'{df_name}_{model_type}_{i+1} \naccuracy: {accuracy_score(y_ts, y_pred)}') 
        for par in params:
            print(f'{par}: {model.get_params()[par]}')
            



#calcolo performance di un modello con le principali misure in sklearn
def performance(model, X_ts, y_true, measure = accuracy_score, multiclass = False, average = None):
  
    y_pred = model.predict(X_ts)
    
    #la misura accuracy_score non ha average come parametro perché agisce in generale e non classe per classe
    supports_average = 'average' in inspect.signature(measure).parameters
    if supports_average:
        
        #se è multiclasse, decidiamo se vogliamo la metrica media o per classe
        #lo facciamo con il parametro average
        if multiclass:
            avg = average 
        else:
            avg = 'binary'
        
        score = measure(y_true, y_pred, average = avg)
    else:
        score = measure(y_true, y_pred)
    
    return score



#calcolo della differenza in performance tra due modelli
def performance_difference(models:list, X_ts, y_true, measure = accuracy_score, multiclass = False, average = None):
        
        scores = [performance(model, X_ts, y_true, measure = measure, multiclass = multiclass, average = average) for model in models]
        
        if len(scores) > 2:
            raise ValueError(f'The function handles at most two models, you are comparing {len(models)} models')
        
        return abs(scores[0] - scores[1])


#prende un modello o una lista di modelli e ne calcola le predizioni e le probabilità per ogni classe
def predictions(models, X_ts, limit_to_two = True):
        
    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]
    
    if limit_to_two and len(models) > 2:
        raise ValueError(f'The function handles at most 2 models, you are comparing {len(models)} models')
    
    preds = np.array([model.predict(X_ts) for model in models])
    probs = np.array([model.predict_proba(X_ts) for model in models])
    
    if len(models) == 1:
        preds = preds[0]
        probs = probs[0]
        
    return preds, probs


#calcola la jaccard tra le predizioni dei modelli per definirne la somiglianza
def predictions_similarity(models, X_ts, average = None, multiclass = False):

    preds, _ = predictions(models, X_ts)
    
    #anche qua con il parametro average scegliamo la jaccard media o per classe
    if multiclass:  
        similarity = jaccard_score(preds[0], preds[1], average = average)
    else:
        similarity = jaccard_score(preds[0], preds[1], average = 'binary')
        
    return similarity


#applica una metrica facendo la differenza tra quelle che rendono tuple e quelle
#che rendono un valore
def apply_metric(x, y, metric): 
    result = metric(x, y) 
    if isinstance(result, tuple): 
        return result[0] 
    else: 
        return result

#filtro per predizioni concordanti e discordanti
def prediction_concordance_filter(models, X_ts, to_filter, concordant = True):
    
    preds, _ = predictions(models, X_ts)
    preds1, preds2 = preds
    condition = (preds1 == preds2) if concordant  else (preds1 != preds2) 
    
    if not isinstance(to_filter, (list, tuple)):
        return to_filter[condition]
    else:
        return [f[condition] for f in to_filter]

#calcola la similarità tra confidence in termini di correlazione ed errore    
def confidence_similarity(models, X_ts, y_true, metric = spearmanr, preds_concordance = True):

    _, probs = predictions(models, X_ts) 
    probs1, probs2 = probs
    
    #filtriamo per predizioni concordanti o discordanti
    m1_probs, m2_probs = prediction_concordance_filter(models, X_ts, (probs1, probs2), concordant = preds_concordance)
    #in caso di classificazione binaria
    if m1_probs.shape[1] == 2:
        
        result = apply_metric(m1_probs[:, 1], m2_probs[:, 1], metric)       
        return result
    
    #in caso multiclasse
    else:
        results = []
        for class_ in range(m1_probs.shape[1]):
            result = apply_metric(m1_probs[:, class_], m2_probs[:, class_], metric)
            
            results.append(result)
        return np.array(results)
    


#sceglie il numero di centroidi adatto basandosi sulla silhouette
def choose_centroids(data, min_centroids, max_centroids):
    best_n = min_centroids
    best_score = -1
    for n in range(min_centroids, max_centroids+1):
        kmeans = KMeans(n_clusters = n, n_init = 10, random_state = 0).fit(data)
        score = silhouette_score(data, kmeans.labels_)
    
        if score > best_score:
            best_score = score
            best_n = n
    return best_n

      
#filtra i centroidi
def kmeans_centroids(data, min_centroids, max_centroids, n_init = 10):
    
    n_clusters = choose_centroids(data, min_centroids, max_centroids)
    kmeans = KMeans(n_clusters = n_clusters, n_init = n_init, random_state = 0).fit(data)
    
    return kmeans.cluster_centers_


#rende gli shap values per le features di un modello
def get_shaps(model, X_ts, min_centroids = 2, max_centroids = 50, expl = KernelExplainer, sampled_results = False):
    
    background = kmeans_centroids(X_ts, min_centroids, max_centroids)
    n_classes = model.predict_proba(X_ts[:1]).shape[1]
    if expl == KernelExplainer:
        if n_classes == 2:
            explainer = expl(model.predict, background)
        else:
            explainer = expl(model.predict_proba, background)
    else:
        explainer = expl(model)
    
    #scegliamo se avere gli shap solo dei centroidi
    if sampled_results:
        shaps = explainer(background).values 
    else:
        shaps = explainer(X_ts).values
    return shaps

#masker = X_ts

#considera la somiglianza dell'influenza delle features instance per instance
#definibile sia con misure di correlazione che di errore
def shaps_similarity(models, X_ts,
                     explainer = KernelExplainer, 
                     metric = pearsonr, 
                     preds_concordance = True,
                     min_centroids = 2, 
                     max_centroids = 50, 
                     sampled_results = False):    
    #filtriamo
    X_ts = prediction_concordance_filter(models, X_ts, X_ts, concordant = preds_concordance)
    
    shaps1 = get_shaps(models[0], X_ts, min_centroids, max_centroids, expl = explainer, sampled_results = sampled_results)
    shaps2 = get_shaps(models[1], X_ts, min_centroids, max_centroids, expl = explainer, sampled_results = sampled_results)

    tot_measures = []
    n_classes = shaps1.shape[2] if shaps1.ndim == 3 else 1
    for i in range(shaps1.shape[0]):
        #multiclasse
        if n_classes > 1:

            instance_measures = []
            for c in range(n_classes): 
                measure = apply_metric(shaps1[i, :, c], shaps2[i, :, c], metric = metric)
                
                instance_measures.append(measure)
            tot_measures.append(instance_measures)
        #binario
        else:
            measure = apply_metric(shaps1[i, :], shaps2[i, :], metric = metric)
            tot_measures.append(measure)
            
    return np.array(tot_measures)



def fairness_explorer(model, X_ts, y_true, sensitive_features, metrics = selection_rate):
    y_pred, _ = predictions(model, X_ts)
    explorer = MetricFrame(
                           metrics = metrics, 
                           y_true = y_true, 
                           y_pred = y_pred,
                           sensitive_features = sensitive_features
                           )
    return explorer.by_group


def model_fairness(model, X_ts, y_true, sensitive_features, metric = demographic_parity_difference):
    
    y_pred, _ = predictions(model, X_ts)
    fairness = metric(
                    y_true = y_true,
                    y_pred = y_pred, 
                    sensitive_features = sensitive_features
                    )
    return fairness


def model_fairness_collection(model, X_ts, y_true, sensitive_features, metrics: dict, return_names = False):
    
    fairness_metrics = {}
    for metric_name, metric in metrics.items():
        fairness = model_fairness(model, X_ts, y_true, sensitive_features, metric)
        fairness_metrics[metric_name] = fairness
        
    return fairness_metrics if return_names else np.array(list(fairness_metrics.values()))

#FAMILY SPECIFIC

#calcola l'altezza del rtc
def tree_depth(node):
    if node is None:
        return 0
    
    left_depth = tree_depth(node["left_node"])
    right_depth = tree_depth(node["right_node"])
    
    return 1 + max(left_depth, right_depth)

def feature_importance_rtc(model, feature_names):
    
    importances = model.compute_feature_importances()
    return pd.Series(importances, index=feature_names)

#trova le features usate e non usate dal modello
def features_usage_rtc(model, feature_names):
    
    importances = feature_importance_rtc(model, feature_names)
    useful = list(importances[importances != 0].index)
    ignored = list(importances[importances == 0].index)
    return {'used_features': useful, 'ignored_features': ignored}





if __name__=='__main__':
    #print(help(ClassificationMetric))
    df_name, df = read_german_credit(basepath = "C:/Users/franc/OneDrive/Desktop/Magistrale/Tesi modelli equivalenti/")
    
    #X_tr, X_ts, y_tr, y_ts = split_ts(df_name, df, save_path = 'C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Split_salvati')
    #model_types = ['knn', 'rtc', 'lr']
    #for model in model_types:
        #train_models(model_type= model, 
                     #df_name = df_name, 
                     #X_tr = X_tr, 
                     #X_ts = X_ts, 
                     #y_tr = y_tr, 
                     #y_ts = y_ts, 
                     #save_path = 'C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Models/german'
                     #)
    
    X_tr = np.genfromtxt('C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt('C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_true = np.genfromtxt('C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    m1 = load('C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_rtc_1.joblib')
    m2 = load('C:/Users/franc/OneDrive/Desktop/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_rtc_1.joblib')
    models = [m1, m2]
    X_ts_df = pd.DataFrame(X_ts, columns=df.columns[:-1])
    
    #corretta
    preds, probs = predictions(models, X_ts)
    print(preds[0])
    
    #corretta
    score = performance(models[0], X_ts, y_true, 
                        measure = f1_score,
                        multiclass = True,
                        average = None)
    print(score)
    
    #corretta
    perf_diff = performance_difference(models, X_ts, y_true,
                                       measure = f1_score,
                                       multiclass = True,
                                       average = 'weighted')
    assert perf_diff == 0, 'le performance dello stesso modello devono essere uguali'
    print(perf_diff)
    
    
    #corretto
    similarity = predictions_similarity(models, X_ts,
                                        average = 'weighted',
                                        multiclass = True)
    assert similarity == 1, 'le performance dello stesso modello devono essere uguali'
    print(similarity)

    print(confidence_similarity(models, X_ts, y_true))
    assert confidence_similarity(models, X_ts, y_true) == 1
    
    x = shaps_similarity(models, X_ts, sampled_results=True)
    print(x)
    
    centr = kmeans_centroids(X_ts, min_centroids = 2, max_centroids = 50)
    print(centr.shape)

    shaps = get_shaps(m1, X_ts, expl = TreeExplainer, sampled_results = True)
    print(shaps.shape)
    print(shaps[1, :, 1])
    
    
    fairness = model_fairness(m1, X_ts, y_true, sensitive_features = X_ts_df['purpose_education'])
    print(fairness)
    explorer = fairness_explorer(m1, X_ts, y_true, sensitive_features = X_ts_df['purpose_education'])
    print(explorer)
    
    fairness_metrics = {
                        'demographic_parity_diff': demographic_parity_difference,
                        'equalized_odds_diff': equalized_odds_difference,
                        'equal_opportunity_diff': equal_opportunity_difference
                        }

    fairness = model_fairness_collection(m1,
                                         X_ts,
                                         y_true, 
                                         sensitive_features = X_ts_df['purpose_education'], 
                                         metrics = fairness_metrics,
                                         return_names = False)
    print(type(fairness))


    rules = m2.get_rules(columns_names=df.columns)
    rules
    print(rules.keys())
    print(rules['node_id'])
    print(rules['left_node']['left_node']['left_node']['node_id'])
    rules = m2.get_rules(columns_names=df.columns)
    tree_depth(rules)
    m2.print_rules(rules, columns_names=df.columns)
    m2.get_predicates()['R']

    help(RuleTreeClassifier())
    
    imp = feature_importance_rtc(m1, feature_names = df.columns[:-1])  
    #print(set(imp[imp !=0].index))
    
    used = features_usage_rtc(m1, feature_names = df.columns[:-1])
    print(used)
    
    a = {1,2,3}
    b= {1,2,5,7}
    print(overlap(a,b))

    sim = neighbours_similarity(models, X_ts)
    print(len(sim))
    assert np.all(sim == 1)
    























