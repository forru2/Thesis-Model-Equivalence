from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import apply_distance_metric
import os
from scipy.stats import pearsonr


#rende un pairplot in cui vediamo le relazioni tra variabili del df di metriche e le correlazioni
def plot_relations(df, metrics_to_plot:list, save = False, file_path = '.', title = '', figsize = (10,10), scale = True, figname = 'Metrics Relations',
                   scaler = MinMaxScaler, zoom_on_rs = False, color_rs = False, labelsize = 40,
                   label_rotation = 0, legendsize = 10, alpha = 0.8, palette = None,
                   show_corr = True, show_xlab = True, show_ylab = True):
    
    plot_df = df.copy()    
    if scale:
        cat_cols = [c for c in ['model_type', 'is_rashomon'] if c in plot_df.columns]
        cat_data = plot_df[cat_cols].copy() if cat_cols else None
    
        sc = scaler().set_output(transform = 'pandas')
        plot_df_scaled = sc.fit_transform(plot_df[metrics_to_plot])
        if cat_data is not None:
            plot_df = pd.concat([plot_df_scaled, cat_data], axis=1).reset_index(drop=True)
        else:
            plot_df = plot_df_scaled

    if zoom_on_rs:
        hue_col = 'model_type'
        palette_map = {'RuleTreeClassifier': 'blue', 'lr': 'red', 'knn': 'goldenrod'}        
    elif color_rs:
        hue_col = 'is_rashomon'
        palette_map = {1: 'blue', 0: 'grey'}
    else:
        hue_col = None
        palette_map = None
            
    p = sns.PairGrid(plot_df, vars=metrics_to_plot, hue=hue_col, palette=palette_map)

    p.map_diag(sns.kdeplot, fill=True, alpha=0.3)
    p.map_lower(sns.scatterplot, s=figsize[0]*6, alpha=alpha)
    
    if hue_col is not None:
        p.add_legend(bbox_to_anchor=(0.9, 0.5), loc='center left')
    
    p.fig.suptitle(title, fontsize = figsize[0]*4, fontweight='bold', y=1.05)
    p.fig.set_size_inches(figsize)

    n = len(metrics_to_plot)
    for k,ax in enumerate(p.axes.flat):
        if ax is None: continue

        i = k // n
        j = k % n
        
        if j > i:
            if show_corr:
                val_x, val_y = plot_df[metrics_to_plot[j]], plot_df[metrics_to_plot[i]]
                r = apply_distance_metric(val_x, val_y, metric=pearsonr)
                color = 'darkred' if r < -0.4 else 'darkblue' if r > 0.4 else 'black'
                
                ax.annotate(f"{r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                            ha='center', va='center', fontsize=figsize[0]*3, fontweight='bold', color=color)
                ax.set_facecolor('#f9f9f9')
            ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

        else:
            if i == n - 1 and show_xlab:
                ax.set_xlabel(metrics_to_plot[j], fontsize=labelsize, rotation=90, labelpad=10)
                ax.set_xticks([0, 1])
                ax.tick_params(axis='x', labelsize=figsize[0]*2, labelbottom=True)
            else:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)

            if j == 0 and show_ylab:
                ax.set_ylabel(metrics_to_plot[i], fontsize=labelsize, rotation=label_rotation, 
                              horizontalalignment='right', labelpad=10)
                ax.set_yticks([0, 1])
                ax.tick_params(axis='y', labelsize=figsize[0]*2, labelleft=True)
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

    if hasattr(p, '_legend') and p._legend is not None:
        plt.setp(p._legend.get_texts(), fontsize = labelsize)
        plt.setp(p._legend.get_title(), fontsize = labelsize, fontweight = 'bold')
        handles = p._legend.legend_handles if hasattr(p._legend, 'legend_handles') else p._legend.legendHandles
        for handle in handles:
            handle.set_markersize(labelsize/2)
            
    if save:
        plt.savefig(os.path.join(file_path, f'{figname}.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()


#fa una griglia di kdeplots per vedere la distribuzione delle metriche per categoria
def plot_variables_distributions(dataframes:list, config_metrics:list, percentile = 90, xmin = None, prefix = 'val',
                                 save = False, file_path = '.', title = '', figname = 'Metrics Distribution per Family'):
    labels = ['RTC', 'KNN', 'LR']
    colors = ['skyblue', 'orange', 'green']

    n_cols = 3 if len(config_metrics) > 1 else 1
    n_rows = (len(config_metrics) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (8.2*n_cols, 5*n_rows), squeeze = False)
    axs = axs.flatten()
    
    for i, metric in enumerate(config_metrics):
        ax = axs[i]
        col = metric
        vals = np.concatenate([df[col].values for df in dataframes])
        threshold = round(np.percentile(vals, percentile), 2)

        for ds, label, color in zip(dataframes, labels, colors):           
            numb = len(ds[ds[col] >= threshold])
            l = f'{numb} {label}'
            sns.kdeplot(ds[col], 
                        fill=True, 
                        color=color, 
                        alpha=0.3,
                        label= l,
                        linewidth=5,
                        ax=ax,
                        clip=(0.0, 1.0)) 

        ax.axvline(threshold, color='red', linestyle='--', linewidth = 2.5)

        y_max = ax.get_ylim()[1]
        ax.text(threshold, y_max * 0.9, f'Top {100-percentile}%', 
                color='red', 
                fontsize=16, 
                fontweight='bold',
                va='top', ha='right', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        if xmin is not None:
            ax.set_xlim(xmin, 1.02)
        if n_cols == 1 and n_rows == 1:
            ax.set_title(title, fontsize = 20, fontweight = 'bold')
        else:
            ax.set_title(col.replace(f'{prefix}_', '').capitalize(), fontsize = 20, fontweight = 'bold')
        ax.set_xlabel(col.replace(f'{prefix}_', '').capitalize(), fontsize=30)
        ax.set_ylabel('Density', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20, loc='best', frameon=True, framealpha=0.5, facecolor='white')
        ax.grid(axis='y', alpha=0.3)
    
    for j in range(len(config_metrics), len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(file_path, f'{title}.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
