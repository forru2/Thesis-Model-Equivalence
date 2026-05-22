import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#mi fa il preprocessing che mi serve sulle variabili
def preprocess_data(df, prefix = 'val'):
    new_df = df.copy()
    new_df['knn_complexity'] = 1-(1/df['n_neighbors']) #intervallo 0, 1 direttamente proporzionale

    for col in [f'{prefix}_equalized_odds', f'{prefix}_cuae',f'{prefix}_dem_parity', f'{prefix}_equal_opportunity', f'{prefix}_predictive_equality',
                f'{prefix}_predictive_parity', f'{prefix}_neg_predictive_parity', f'{prefix}_expected_calibration_error', f'{prefix}_cat_cross_entropy_on_gt']:
        if col == f'{prefix}_cat_cross_entropy_on_gt':
            new_df[f'{col}_inverted'] = np.exp(- df[col])
        else:
            new_df[f'{col}_inverted'] = 1 - df[col].abs() #intervallo 0, 1 direttamente proporzionale
    return new_df

#mi da le variabili rappresentative per categoria (pc1)
def get_pca_metrics(df, n_components = 1, prefix = 'val'):
    scaler = StandardScaler().set_output(transform = 'pandas')
    numeric_cols = df.select_dtypes('number').columns
    scaled_df = df.copy()
    scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    pca = PCA(n_components = n_components)
    
    performance = pca.fit_transform(scaled_df[[f'{prefix}_accuracy', f'{prefix}_f1', f'{prefix}_precision', f'{prefix}_recall']].values)[:,0]
    perf_components = pca.components_[0]
    if sum(perf_components) < 0:
        performance = -performance
        
    fairness = pca.fit_transform(scaled_df[[f'{prefix}_equalized_odds_inverted', f'{prefix}_cuae_inverted', f'{prefix}_dem_parity_inverted', f'{prefix}_equal_opportunity_inverted',
                                            f'{prefix}_predictive_equality_inverted', f'{prefix}_predictive_parity_inverted', 
                                            f'{prefix}_neg_predictive_parity_inverted']].values)[:,0]
    fair_components = pca.components_[0]
    if sum(fair_components) < 0:
        fairness = -fairness
        
    fair_robustness = pca.fit_transform(scaled_df[[f'{prefix}_dem_parity_robustness', f'{prefix}_equal_opp_robustness', 
                                                   f'{prefix}_predictive_eq_robustness',
                                                   f'{prefix}_pred_parity_robustness', f'{prefix}_neg_pred_parity_robustness',
                                                   f'{prefix}_eq_odds_robustness']].values)[:,0]
    fr_components = pca.components_[0]
    if sum(fr_components) < 0:
        fair_robustness = -fair_robustness
        
    perf_robustness = pca.fit_transform(scaled_df[[f'{prefix}_acc_robustness', f'{prefix}_f1_robustness', 
                                                   f'{prefix}_precision_robustness', f'{prefix}_recall_robustness']].values)[:,0]
    pr_components = pca.components_[0]
    if sum(pr_components) < 0:
        perf_robustness = -perf_robustness
    
    reliability = pca.fit_transform(scaled_df[[f'{prefix}_expected_calibration_error_inverted', f'{prefix}_cat_cross_entropy_on_gt_inverted', 
                                                   f'{prefix}_mean_confidence_on_gt']].values)[:,0]
    rel_components = pca.components_[0]
    if sum(rel_components) < 0:
        reliability = -reliability
    
    complexity = np.zeros(len(df))
    rtc_mask = df['model_type'] == 'RuleTreeClassifier'
    knn_mask = df['model_type'] == 'knn'
    lr_mask  = df['model_type'] == 'lr'
    
    rtc_cols = [f'{prefix}_rtc_depth', f'{prefix}_rtc_n_nodes', f'{prefix}_rtc_n_rules']
    complexity[rtc_mask] = pca.fit_transform(scaled_df.loc[rtc_mask, rtc_cols].values)[:, 0] 
    #print(pca.components_[0])
    complexity[knn_mask] = scaled_df.loc[knn_mask, 'knn_complexity'].values       
    complexity[lr_mask] = scaled_df.loc[lr_mask, f'{prefix}_coeff_magnitude'].values
    
    d = {
        'performance': performance, 
        'perf_robustness': perf_robustness,
        'fairness': fairness, 
        'fair_robustness': fair_robustness,  
        'complexity': complexity,
        'reliability': reliability,
        'model_id': df['model_id'].values,
        'model_type': df['model_type'].values
    }
        
    return pd.DataFrame(d)