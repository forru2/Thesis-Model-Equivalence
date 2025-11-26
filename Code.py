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
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr, pearsonr
import random as rd
from joblib import dump, load
from shap import TreeExplainer, LinearExplainer, KernelExplainer
#import os
import inspect
from fairlearn.metrics import selection_rate, MetricFrame, true_positive_rate, false_positive_rate
#from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equal_opportunity_difference, equal_opportunity_ratio, equalized_odds_difference, equalized_odds_ratio
#from fairlearn import metrics
#from aif360.metrics import ClassificationMetric
#from aif360.datasets import BinaryLabelDataset

#split df and save
def split_ts(df_name, df, save = True, scale = True, test_size = 0.3, save_path = './'):
    
    y = df['y'].values
    X = df[df.columns[:-1]].values
    
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = test_size, random_state = 42, stratify = y)
    
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_ts = scaler.transform(X_ts)
    
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
        val = result[0] 
    else: 
        val = result
    #per gestire la divisione per 0 in caso di correlazione   
    if np.isnan(val):
        std_x = np.std(x)
        std_y = np.std(y)
        
        if std_x == 0 and std_y == 0:
            return 1.0
        else:
            return 0.0
    return val

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
#se concordano/discordano, quanto sono simili in sicurezza?   
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
    
    #perché il kernel non è deterministico (stima Monte Carlo)
    np.random.seed(0)
    
    n_classes = model.predict_proba(X_ts[:1]).shape[1]
    if expl == KernelExplainer:
        
        background = kmeans_centroids(X_ts, min_centroids, max_centroids)
        if n_classes == 2:
            explainer = expl(model.predict, background)
        else:
            explainer = expl(model.predict_proba, background)
    elif expl == TreeExplainer:
        explainer = expl(model)
    
    elif expl == LinearExplainer:
        explainer = expl(model, masker = X_ts)
    
    shaps = explainer(X_ts).values
    return shaps


#calcola la similarità tra due array delle importanze
#la uso sia in shaps_similarity che in feature_importance_similarity_rtc
def importance_array_similarity(arr1, arr2, metric):
    tot_measures = []
    n_classes = arr1.shape[2] if arr1.ndim == 3 else 1
    for i in range(arr1.shape[0]):
        #multiclasse
        if n_classes > 1:

            instance_measures = []
            for c in range(n_classes): 
                measure = apply_metric(arr1[i, :, c], arr2[i, :, c], metric = metric)
                
                instance_measures.append(measure)
            tot_measures.append(instance_measures)
        #binario
        else:
            measure = apply_metric(arr1[i, :], arr2[i, :], metric = metric)
            tot_measures.append(measure)
            
    return np.array(tot_measures)


#considera la somiglianza dell'influenza delle features instance per instance in caso di previsioni concord/discord
#definibile sia con misure di correlazione che di errore
def shaps_similarity(models, X_ts,
                     explainer = KernelExplainer, 
                     metric = spearmanr, 
                     preds_concordance = True,
                     min_centroids = 2, 
                     max_centroids = 50, 
                     sampled_results = False):    
    #filtriamo
    X_ts = prediction_concordance_filter(models, X_ts, X_ts, concordant = preds_concordance)
    
    imp1 = get_shaps(models[0], X_ts, min_centroids, max_centroids, expl = explainer, sampled_results = sampled_results)
    imp2 = get_shaps(models[1], X_ts, min_centroids, max_centroids, expl = explainer, sampled_results = sampled_results)
       
    return importance_array_similarity(imp1, imp2, metric = metric)

#FAIRNESS

def negative_predictive_value(y_true, y_pred):     
    tn, _, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() 
    if (tn + fn) == 0:
        return 0.0
    return tn / (tn + fn)


#crea una serie pandas con metriche e rispettivi valori
def fairness_explorer(model, X_ts, y_true, sensitive_features, metrics: dict):
   
    y_pred = model.predict(X_ts)
    classes = np.unique(y_true)
    
    if len(classes) > 2:
        fairness_per_class = []
        for cl in classes:
            y_true_binary = (y_true == cl).astype(int)
            y_pred_binary = (y_pred == cl).astype(int)
            explorer = MetricFrame(
                                   metrics = metrics, 
                                   y_true = y_true_binary, 
                                   y_pred = y_pred_binary,
                                   sensitive_features = sensitive_features
                                   )
            fairness_per_class.append(explorer.difference())
        return fairness_per_class
    
    else:
        return explorer.difference()

#fairness_explorer()
#mi calcola eo e cuae come fa fairlearn, anche se possono essere visti già da fairness explorer
def advanced_fairness(model, X_ts, y_true, sensitive_features):
    differences = fairness_explorer(model,
                                    X_ts,
                                    y_true,
                                    sensitive_features,
                                    metrics = {
                                        'equal opportunity(TPR)': true_positive_rate,
                                        'predictive equality(FPR)': false_positive_rate,
                                        'predictive parity(precision)': precision_score,
                                        'negative predictive parity(NPV)': negative_predictive_value
                                        })
    
    eq_odds = max(differences['equal opportunity(TPR)'], differences['predictive equality(FPR)'])
    cuae = max(differences['predictive parity(precision)'], differences['negative predictive parity(NPV)'])
    
    return {'equalized_odds': eq_odds, 'cond_use_accuracy_equality': cuae}


fairness_metrics = {
    'demographic parity(selection_rate)': selection_rate,
    'equal opportunity(TPR)': true_positive_rate,
    'predictive equality(FPR)': false_positive_rate,
    'predictive parity(precision)': precision_score,
    'negative predictive parity(NPV)': negative_predictive_value}



#FAMILY SPECIFIC

#mi rende il numero di vicini
def knn_local_complexity(model):
    return model.n_neighbors()

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

#calcola l'importanza delle features instance per instance come shap
def feature_importance_rtc(model, X_ts, multiclass = False):
    if multiclass:
        importances = model.local_interpretation(X_ts)[2]
    else:
        importances = model.local_interpretation(X_ts)[2][:, :, 1]
    
    return importances  

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

#stessa cosa ma per lr
def feature_importance_lr(model, X_ts, multiclass = False):
    #weights (n_classes, n_features), X_ts (n_instances, n_features)
    weights = model.coef_ 
    
    if multiclass:
        unordered_fc = X_ts[:, None, :] * weights[None, :, :]
        feature_contributions = np.transpose(unordered_fc, (0, 2, 1))
        
    else:
        feature_contributions = X_ts * weights
        
    return feature_contributions


#calcola la similarità nele feature importances specifiche per rtc e lr, date le stesse previsioni    
#model type possibili sono ['rtc', 'lr']    
def feature_importance_similarity(models, X_ts, model_type = 'rtc',
                                      preds_concordance = True, 
                                      multiclass = False, 
                                      metric = spearmanr
                                      ):
    X_ts = prediction_concordance_filter(models, X_ts, to_filter = X_ts, concordant = preds_concordance)
    
    if model_type == 'rtc':
        imp1 = feature_importance_rtc(models[0], X_ts, multiclass = multiclass)
        imp2 = feature_importance_rtc(models[1], X_ts, multiclass = multiclass)
    
    elif model_type == 'lr':
        imp1 = feature_importance_lr(models[0], X_ts, multiclass = multiclass)
        imp2 = feature_importance_lr(models[1], X_ts, multiclass = multiclass)
    
    return importance_array_similarity(imp1, imp2, metric = metric)




if __name__=='__main__':
    #print(help(ClassificationMetric))
    df_name, df = read_german_credit(basepath = "C:/Users/franc/OneDrive/Magistrale/Tesi modelli equivalenti/")
    
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
    #binary
    X_tr = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_X_ts.csv', delimiter=',', skip_header=1)
    y_true = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/german_credit_y_ts.csv', delimiter=',', skip_header=1)
    m1 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_lr_3.joblib')
    m2 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/german/german_credit_lr_3.joblib')
    
    models = [m1, m2]
    X_ts_df = pd.DataFrame(X_ts, columns=df.columns[:-1])
    sensitive_features = X_ts_df['personal_status']
    
    #multiclass
    df_name, df = read_vehicle(basepath = "C:/Users/franc/OneDrive/Magistrale/Tesi modelli equivalenti/")
    
    X_tr = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_X_tr.csv', delimiter=',', skip_header=1)
    X_ts = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_X_ts.csv', delimiter=',', skip_header=1)
    y_true = np.genfromtxt('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Split_salvati/vehicle_y_ts.csv', delimiter=',', skip_header=1)
    m1 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/vehicle/vehicle_rtc_1.joblib')
    m2 = load('C:/Users/franc/OneDrive/Magistrale/Thesis-Model-Equivalence/Models/vehicle/vehicle_rtc_1.joblib')
   
    models = [m1, m2]
    X_ts_df = pd.DataFrame(X_ts, columns=df.columns[:-1])
    sensitive_features = X_ts_df['CIRCULARITY']
    
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
    
    imp1 = get_shaps(m1, X_ts)
    imp2 = get_shaps(m2, X_ts)
    importance_array_similarity(imp1, imp2, metric= spearmanr)
    
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
    print(rules['feature_idx'])
    print(rules['feature_name'], rules['threshold'])
    print(rules['left_node']['left_node']['right_node']['node_id'])
    rules = m2.get_rules(columns_names=df.columns)
    
    tree_depth(rules)
    count_nodes_rtc(rules)
    print(tree_complexity(m2, rules))
    
    m2.print_rules(rules, columns_names=df.columns)
    m2.get_leaf_nodes()

    help(RuleTreeClassifier())
    print(inspect.getsource(RuleTreeClassifier._predict))
    
    for el in m2.local_interpretation(X_ts):
        print(el.shape)
    
    m2.local_interpretation(X_ts)[2][1]
    len(m2.local_interpretation(X_ts))

    imp = feature_importance_rtc(m2, X_ts, feature_names = df.columns[:-1])
    print(imp)
    imp.shape
    
    shap = get_shaps(m1, X_ts)
    print(shap)


    sim = shaps_similarity(models, X_ts, metric = pearsonr)
    sim2 = feature_importance_similarity(models, X_ts, model_type = 'rtc', multiclass = True)
    sim3 = feature_importance_similarity(models, X_ts, model_type = 'lr', multiclass = True)   
    print(sim3.shape)
    
    print(sim2)
    sim.shape
    sim2.shape
    imp = feature_importance_rtc(m2, X_ts, multiclass = True)
    imp.shape
    
    coef = m2.coef_
    print(coef)
    print(coef.shape)
    
    fi = feature_importance_lr(m2, X_ts, multiclass = True) 
    print(fi.shape)      
        
    compl = lr_complexity(m1, multiclass = True)
    print(compl)   

    idx = m2.apply(X_ts)
    print(range(idx.shape[0]))
    lengths = path_lengths(m2, X_ts)
    print(lengths)





















#se entrambi sbagliano, quanto sono simili negli errori in caso multiclasse?

               
#filtra per concordanza/discordanza tra previsioni di sue modelli
#e concordanza/discordanza delle previsioni di entrambi i modelli con la ground truth
def prediction_concordance_filter(models, X_ts, to_filter, y_true = None, concordant = True):
    
    preds, _ = predictions(models, X_ts)
    preds1, preds2 = preds
    
    if y_true is None:
        
        condition = (preds1 == preds2) if concordant  else (preds1 != preds2) 
    else:
        condition = (preds1 == y_true) & (preds2 == y_true) if concordant else (preds1 != y_true) & (preds2 != y_true)
    
    if not isinstance(to_filter, (list, tuple)):
        return to_filter[condition]
    else:
        return [f[condition] for f in to_filter]

  
#se entrambi hanno torto/ragione, quanto sono convinti della ground truth?
def true_class_probability(models, X_ts, y_true):

    _, probs = predictions(models, X_ts)
    probs1, probs2 = probs
    probs1, probs2, y_true = prediction_concordance_filter(models, X_ts, to_filter = (probs1, probs2, y_true), y_true = y_true)
    filtered_probs = probs1, probs2
    
    indexes = y_true.ravel().astype(int)
    probs_true_class = [prob[np.arange(prob.shape[0]), indexes] for prob in filtered_probs]
    
    return np.column_stack((probs_true_class[0], probs_true_class[1]))


out = true_class_probability(models, X_ts, y_true)
print(out)
print(out.shape)












