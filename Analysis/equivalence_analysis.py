from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import os
import numpy as np
from pathlib import Path 
from joblib import load, Parallel, delayed, dump
from Utils import predictions
from itertools import combinations
from feature_importance import importance_array_similarity
from performance import predictions_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, hamming_loss, mean_absolute_error

#prende le metriche che mi interessano dal df e ne calcola le matrici di distanza
#rende un dizionario che contiene le matrici per ogni metrica
#cols è la lista di colonne delle metriche che ci interessano
def get_dist_matrices_from_df(df, cols, dist_metric = 'euclidean', scale = False, scaler = MinMaxScaler):
    ids = df['model_id'].values
    dist_matrices = {}
    for col in cols:
        dist_vec = pdist(df[[col]], metric = dist_metric)
        if scale:
            sc = scaler()
            dist_vec = sc.fit_transform(dist_vec.reshape(-1, 1)).ravel() 
        dist_matrix = squareform(dist_vec)
        dist_matrices[f'{col}_dist({dist_metric})'] = pd.DataFrame(dist_matrix, index = ids, columns = ids)
    return dist_matrices


#rende un dizionario di modelli caricati, ottenibili per model_id
def load_models(model_ids, path = '.'):
    models_dict = {}
    for model_id in model_ids:
        search_pattern = f"**/{model_id}.joblib"
        files_found = list(Path(path).rglob(search_pattern))
        if files_found:
            models_dict[model_id] = load(files_found[0])
    return models_dict


#rende un dizionario con i valori delle preds/probs per ogni model_id
def compute_preds_probs_dict(models_dict, X_ts, save = False, file_path = '.', file_name = '', probs = False):
    model_ids = list(models_dict.keys())
    models = list(models_dict.values())
    if probs:
        result = predictions(models, X_ts, limit_to_two = False)[1]
    else:
        result = predictions(models, X_ts, limit_to_two = False)[0]

    res_dict = dict(zip(model_ids, result))
    if save:
        dump(res_dict, os.path.join(file_path, f'{file_name}.joblib'))
    return res_dict


#prende le funzioni di similarità già calcolate altrove e rende un dizionario col nome della metrica e la matrice di distanza
#le matrici calcolabili sono quella degli shap e quella delle previsioni
def get_dist_matrix_from_arrays(arrays_dict, function, X_ts, metric_name, dist,
                                  save = False, file_path = '.', file_name = '',
                                  scale = False, scaler = MinMaxScaler, **kwargs):
    model_ids = list(arrays_dict.keys())
    n_models = len(model_ids)
    
    dist_vec = []
    for i, j in combinations(range(n_models), 2):
        id_i, id_j = model_ids[i], model_ids[j]
        
        if function is importance_array_similarity:
            res = function(arrays_dict[id_i], arrays_dict[id_j], aggregate = True, metric = dist)
        elif function is predictions_similarity:
            res = function(arrays_dict[id_i], arrays_dict[id_j], metric = dist) 
        dist_vec.append(res)

    dist_vec = np.array(dist_vec)
    if scale:
        sc = scaler()
        dist_vec = sc.fit_transform(dist_vec.reshape(-1, 1)).ravel()        
         
    dist_matrix = {metric_name: pd.DataFrame(squareform(dist_vec), index = model_ids, columns = model_ids)} 
    if save:
        pd.to_pickle(dist_matrix, os.path.join(file_path, f'{file_name}.pkl'))
    return dist_matrix

#prende il dizionario delle matrici di distanza, ne prende la metà superiore e rende un dizionario {metrica: metà superiore}
def get_upper_matrices(dist_matrices):
    upper_dists = {}
    for metric, matrix in dist_matrices.items():
        upper_indices = np.triu_indices_from(matrix, k=1)
        distances = matrix.to_numpy()[upper_indices]
        upper_dists[metric] = distances
    return upper_dists

#normalizza le matrici di distanza con uno scaler a scelta tra quelli di sklearn (MinMax, StandardScaler, RobustScaler)
def scale_matrices(dist_matrices, scaler = MinMaxScaler):
    upper_matrices = get_upper_matrices(dist_matrices)
    ids = dist_matrices[list(dist_matrices.keys())[0]].index
    scaled_matrices = {}
    for metric, matrix in upper_matrices.items():
        sc = scaler()
        scaled_matrix = squareform(sc.fit_transform(matrix.reshape(-1, 1)).flatten())
        scaled_matrices[metric] = pd.DataFrame(scaled_matrix, index = ids, columns = ids)
    return scaled_matrices


#fa gli istogrammi delle matrici di distanza per vedere come sono distribuite
def plot_dist_matrices_distributions(dist_matrices1,
                                     bins = 10, percentiles:list = [10,20,30], cols = 3, figsize = (24,8), xlim = False, xlim_max = 1.75, ylim_max = 40, save = False,
                                     file_path = '.', filename = 'distance matrices distributions comparison', labelsize = 12):

    upper_dists1 = get_upper_matrices(dist_matrices1)
    metrics = list(upper_dists1.keys())
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        dists1 = upper_dists1[metric]
        
        sns.histplot(dists1, ax=ax, bins=bins, color='skyblue', 
                     stat='percent', alpha=0.6)

        colors = ['red','blue','green']
        for p,c in zip(percentiles, colors):
            perc1 = np.percentile(dists1, p)
            ax.axvline(perc1, color=c, linestyle='--', linewidth = 2.5,
                       label=f'{p}%')

        #ax.set_title(f"{metric}", fontweight='bold')
        ax.set_xlabel(metric, fontsize = labelsize, fontweight = 'bold')
        ax.xaxis.set_label_position('top')
        if i % cols == 0:
            ax.set_ylabel('Percent', fontsize = labelsize)
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        
        ax.tick_params(axis='both', which='major', labelsize = labelsize*0.75)

        ax.set_ylim(0, ylim_max)
        if xlim:
            ax.set_xlim(0, xlim_max)  
        if i == 0:
            ax.legend(title = 'Thresholds', title_fontsize = 25, fontsize=25, loc='best', framealpha=0.9)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    if save:
        plt.savefig(os.path.join(file_path, f'{filename}.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()


#prende il dizionario di matrici di distanza {metrica: matrice}
#rende un dizionario uguale ma con le matrici binarie, con 1 nelle caselle che superano il threshold di similarità
#la sognia è definita usando dall'n percentile (default 10) su ogni matrice di distanza
def find_equivalence_from_dist_matrices(dist_matrices:dict, threshold = 10):
    upper_matrices = get_upper_matrices(dist_matrices)
    bin_matrices = {}
    for metric, matrix in dist_matrices.items():
        vals = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
        thr = np.percentile(upper_matrices[metric], threshold)
        bin_matrix = (vals <= thr).astype(int)
        bin_matrices[metric] = bin_matrix
    return bin_matrices

#prende il dizionareio delle matrici di distanza {metrica: matrice}, le trasforma in binarie con la funzione precedente 
#e rende la matrice di consenso
def get_consensus_matrix(dist_matrices:dict, perc = 10, negative = True):
    bin_matrices = find_equivalence_from_dist_matrices(dist_matrices, perc)
    consensus_matrix = sum(bin_matrices.values())
    if negative:
        return len(bin_matrices) - consensus_matrix
    return consensus_matrix

#applica un clustering gerarchico e rende il linkage e le labels
#il linkage serve eventualmente per fare il plot del dendrogramma
#cut_level è il livello di dissenso max di due modelli per finire nello stesso cluster
def get_cluster_info(neg_consensus_matrix, method = 'complete', cut_level = 0, min_cluster_size = 2):
    vals = neg_consensus_matrix if isinstance(neg_consensus_matrix, np.ndarray) else neg_consensus_matrix.values
    clustering = linkage(squareform(vals), method = method)
    labels = fcluster(clustering, t = cut_level, criterion = "distance")
    label_counts = pd.Series(labels).value_counts()
    max_equiv_clusters = len(label_counts[label_counts.values >= 2])

    n_labels = len(label_counts)
    if 1 < n_labels < len(labels):
        sil = silhouette_score(vals, labels, metric = 'precomputed')
    else:
        sil = np.nan
    
    print(f'LABEL COUNTS: \n{label_counts[label_counts.values >= min_cluster_size]}')

    d = {
        'clustering': clustering,
        'labels': labels,
        'label_counts': label_counts,
        'max_equiv_clusters': max_equiv_clusters,
        'silhouette': sil
    }
    return d

#filtra il df delle metriche per prendere solo i modelli dei cluster che ci servono
#min_cluster_counts indica il numero minimo di modelli in un cluster perché venga considerato valido
def get_cluster_metrics_df(df, labels, min_cluster_counts = 2):
    df = df.copy()
    df.loc[:, 'cluster'] = labels
    cluster_counts = df['cluster'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cluster_counts].index

    return df[df['cluster'].isin(valid_clusters)]


def plot_clusters(metrics_per_cluster, df_scaling, metrics_to_plot:list, label_counts:pd.Series, scale = True, 
                  valuable_clusters = None, save = False, file_path = '.', title = '', show_legend = True):

    df_plot = metrics_per_cluster.copy()
    if scale:
        #scaler = MinMaxScaler().set_output(transform = 'pandas')
        #scaled = scaler.fit_transform(metrics_per_cluster[metrics_to_plot])
        #metrics_per_cluster = pd.concat([scaled, metrics_per_cluster[['model_id', 'model_type', 'cluster']]], axis = 1)
        for col in metrics_to_plot:
            v_min, v_max = df_scaling[col].min(), df_scaling[col].max()
            df_plot[col] = (df_plot[col] - v_min) / max(v_max - v_min, 1e-9)
    
    
    if valuable_clusters is not None:
        df_plot = df_plot[df_plot['cluster'].isin(valuable_clusters)]
    
    df_mean = df_plot.groupby('cluster')[metrics_to_plot].mean()
    df_min = df_plot.groupby('cluster')[metrics_to_plot].min()
    df_max = df_plot.groupby('cluster')[metrics_to_plot].max()
    
    categories = list(df_mean.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    colors = plt.get_cmap('tab10', len(df_mean))    
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    for i, (index, row) in enumerate(df_mean.iterrows()):
        color = colors(i)
        
        values = row.values.flatten().tolist()
        values += values[:1] 
        
        lower_bound = df_min.loc[index].values.flatten().tolist()
        lower_bound += lower_bound[:1]        
        upper_bound = df_max.loc[index].values.flatten().tolist()
        upper_bound += upper_bound[:1]

        ax.fill(angles + angles[::-1], 
            upper_bound + lower_bound[::-1], 
            color=color, alpha=0.2)
    
        model_types = np.unique(df_plot['model_type'][df_plot['cluster'] == index].values)
        label = fr'$\mathbf{{Cl\_{index}:}}$ size: {label_counts[index]}, types: {model_types}'
        ax.plot(angles, values, linewidth=2, linestyle='solid', label = label, color = color)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize = 15)
    
    if scale:
        ax.set_ylim(-0.1, 1.1)

    if show_legend:
        plt.legend(loc = 'upper right', bbox_to_anchor = (2.2, 1.1), fontsize = 13)
    plt.title(title, size = 15, y = 1.1, fontweight='bold')
    
    if save:
        plt.savefig(os.path.join(file_path, f'{title}.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_neg_consensus_matrix(neg_consensus_matrix, figsize = (10, 10), save = False, file_path = '.', filename = 'Consensus Matrix Clustermap'):
    p = sns.clustermap(neg_consensus_matrix, method = 'complete', figsize = figsize)
    if save:
        plt.savefig(os.path.join(file_path, f'{filename}.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()

def plot_shaps(shap_values_list, feature_names, top_n = 10, title = 'SHAP Feature Importance & Model Consistency',
               save = False, file_path = '.'):    
    all_importances = []
    for sv in shap_values_list:
        vals = sv.values if hasattr(sv, 'values') else sv
        global_importance = np.abs(vals).mean(axis=0)
        all_importances.append(global_importance)
    
    df_models = pd.DataFrame(all_importances, columns=feature_names)
    summary_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': df_models.mean(axis=0),
        'std_dev': df_models.std(axis=0)
    })
    
    summary_df = summary_df.sort_values(by='mean_importance', ascending = True)
    means = summary_df['mean_importance'][-top_n:]
    stds = summary_df['std_dev'][-top_n:]
    error = [np.zeros_like(stds), stds]
    
    plt.figure(figsize=(10, 4))
    bars = plt.barh(summary_df['feature'][-top_n:], summary_df['mean_importance'][-top_n:], 
                    xerr = error, 
                    color = 'steelblue', edgecolor='navy', 
                    capsize = 5, 
                    error_kw = {'elinewidth': 2, 'capthick': 2},
                    alpha = 0.8)

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=10)
    plt.xlabel('Mean(|SHAP|)', fontsize = 15)
    plt.title(title, fontsize = 17, fontweight = 'bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(file_path, f'{title}.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()

def mean_rel_abs_error(labels_pred, labels_true):
    output_domain = np.unique(labels_true)
    errors = []
    for lab in output_domain:
        pk_pred = (labels_pred == lab).mean()
        pk_true = (labels_true == lab).mean()
        margin = np.abs(pk_pred - pk_true)/pk_true
        errors.append(margin)
    mean_rel_abs_error = np.mean(errors)
    return mean_rel_abs_error

def rel_abs_error_cluster(models:list, X_ts, y_ts):
    mae_list = []
    for m in models:
        labels_pred,_ = predictions(m, X_ts)
        labels_true = y_ts
        mae = mean_rel_abs_error(labels_pred, labels_true)
        mae_list.append(mae)
    mean = np.mean(mae_list)
    std = np.std(mae_list)
    return mean, std


def get_class_frequency(preds_list, num_classes = 2):
    tot_perc = []
    for preds in preds_list:
        counts = np.bincount(preds.astype(int), minlength = num_classes)
        perc = counts / len(preds)
        tot_perc.append(perc)    
    
    mean_class_freq = np.mean(tot_perc, axis = 0)
    std_class_freq = np.std(tot_perc, axis = 0)
    return mean_class_freq, std_class_freq



def stats_class_freq_cluster(metrics_per_cluster, preds_dict, num_classes = 2):
    mean_freq_per_cluster = {}
    std_freq_per_cluster = {}
    for cl in np.unique(metrics_per_cluster['cluster'].values):
        cluster_ids = metrics_per_cluster['model_id'][metrics_per_cluster['cluster'] == cl].values
        filtered_preds = [preds_dict[m_id] for m_id in cluster_ids if m_id in preds_dict]
        if filtered_preds:
            mean_class_freq, std_class_freq = get_class_frequency(filtered_preds, num_classes)
            mean_freq_per_cluster[cl] = mean_class_freq
            std_freq_per_cluster[cl] = std_class_freq
    return mean_freq_per_cluster, std_freq_per_cluster

def mean_freq_classes_per_cluster(metrics_per_cluster, preds_dict):
    mean_freq, std_freq = stats_class_freq_cluster(metrics_per_cluster, preds_dict)

    df_means = pd.DataFrame(mean_freq)
    df_stds = pd.DataFrame(std_freq)
    return df_means, df_stds