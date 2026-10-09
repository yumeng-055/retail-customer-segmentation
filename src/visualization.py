"""Export readable aggregate figures in English."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from .config import CATEGORIES

PALETTE = ['#577590', '#43AA8B', '#F8961E', '#9B5DE5']
SEGMENT_COLORS = dict(zip(['Low-engagement shoppers', 'Family-focused shoppers',
                          'Fashion and technology shoppers', 'High-income omnichannel shoppers'], PALETTE))
SHORT = {'monthly_income':'Income (SGD/month)', 'spend_grocery':'Grocery (SGD/month)',
         'spend_fashion':'Fashion (SGD/month)', 'spend_electronics':'Electronics (SGD/month)',
         'spend_dining':'Dining (SGD/month)', 'spend_online':'Online (SGD/month)',
         'visits_store_per_month':'Store visits/month', 'visits_online_per_month':'Online visits/month',
         'age':'Age (years)', 'num_children':'Children', 'loyalty_score':'Loyalty score'}

def export_figures(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.0)
    plt.rcParams.update({'savefig.dpi':220, 'font.family':'DejaVu Sans'})

    def save(name):
        plt.tight_layout()
        plt.savefig(output / f'{name}.png', bbox_inches='tight')
        plt.close()

    quality = result['tables']['data_quality']
    quality = quality[quality.missing_count > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh([SHORT.get(c,c) for c in quality.field], quality.missing_count, color=PALETTE[0])
    axes[0].set(xlabel='Missing cells (2,000 customers)', title='50 missing values per field (2.5%)')
    axes[1].axis('off')
    axes[1].text(0, .9, 'Median: income and children\nMean: electronics and online spending\n\nlog1p: income and store visits\nStandardScaler: eight model features\n\nOriginal units kept for profiles\nNo rows or extremes removed', va='top', fontsize=12)
    save('data_quality')

    v = result['tables']['pca_variance']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(v.PC, v.explained_variance, color=PALETTE[0])
    axes[0].set(title='Variance by component', ylabel='Explained variance share')
    axes[1].plot(range(1,9), v.cumulative_variance, 'o-', color=PALETTE[1])
    axes[1].axvline(5, color='gray', linestyle='--', label='Five components retained')
    axes[1].set(title='Cumulative retained variance', xlabel='Components', ylabel='Cumulative variance share', ylim=(0,1.05))
    axes[1].legend(fontsize=9)
    save('pca_variance')

    plt.figure(figsize=(11, 5))
    w = result['tables']['pca_weights'].rename(index=SHORT)
    sns.heatmap(w, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label':'Component weight'})
    plt.title('PCA component weights (components_.T)')
    save('pca_weights')

    for method, name in [('kmeans','kmeans_candidates'), ('ward','ward_candidates')]:
        t = result['tables'][name]
        cols = ['inertia','silhouette','CHI','DBI'] if method=='kmeans' else ['silhouette','CHI','DBI','minClusterRatio']
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col in zip(axes.flat, cols):
            ax.plot(t.K, t[col], 'o-', color=PALETTE[0])
            ax.axvline(4, linestyle='--', color='gray')
            ax.set(xlabel='K', ylabel=col, title=col)
        fig.suptitle('K-Means: silhouette uses 1,000 sampled customers' if method=='kmeans' else 'Ward: silhouette uses all 2,000 customers')
        save(name)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, method in enumerate(['kmeans','ward']):
        t = result['tables'][f'{method}_stability']
        for col, metric in enumerate(['ARI','NMI']):
            ax = axes[row,col]
            ax.errorbar(t.K,t[f'{metric}_mean'],yerr=t[f'{metric}_std'],fmt='o-',capsize=3,color=PALETTE[row])
            ax.set(title=f'{method.title()} subsampling {metric}', xlabel='K', ylabel='Mean +/- population SD')
    fig.suptitle('Different reference protocols; agreement is not ground-truth accuracy')
    save('stability')

    t = result['tables']['kmeans_stability']
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for ax, metric in zip(axes,['ARI','NMI']):
        ax.errorbar(t.K,t[f'{metric}_init_mean'],yerr=t[f'{metric}_init_std'],fmt='o-',capsize=3,color=PALETTE[0])
        ax.set(title=f'Initialization stability: {metric}',xlabel='K',ylabel='Mean +/- population SD')
    save('initialization_stability')

    z = result['linkage']
    cut = (z[-3,2] + z[-4,2]) / 2
    plt.figure(figsize=(12, 5))
    dendrogram(z, no_labels=True, color_threshold=cut)
    plt.axhline(cut, color='#D1495B', linestyle='--', label='Four-cluster cut')
    plt.xlabel('Leaves ordered by tree (all 2,000 customers)')
    plt.ylabel('Ward merge height in five-PC space')
    plt.title('Full Euclidean/Ward dendrogram')
    plt.legend()
    save('ward_dendrogram')

    t = result['tables']['model_comparison']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes,['silhouette','CHI','DBI']):
        ax.bar(t.model, t[metric], color=PALETTE[:2])
        ax.set(title=metric)
        ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('K=4 internal metrics; silhouette sampling differs by method')
    save('model_comparison')

    profiles = result['profiles']['k4']
    names = result['names']
    colors = {i: SEGMENT_COLORS[names[i]] for i in profiles['sizes'].index}
    labels = [names[i].replace(' shoppers','\nshoppers') for i in profiles['sizes'].index]
    plt.figure(figsize=(11, 4))
    bars = plt.bar(labels,profiles['sizes']['count'],color=[colors[i] for i in profiles['sizes'].index])
    for bar, (_, row) in zip(bars, profiles['sizes'].iterrows()):
        plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+10,f"{row['count']:.0f} ({row['share']:.1%})",ha='center')
    plt.ylim(0, profiles['sizes']['count'].max()*1.2)
    plt.ylabel('Customers (denominator: 2,000)')
    plt.title('Final four customer segments')
    save('segment_sizes')

    medians = profiles['medians']
    standardized = (medians-medians.mean())/medians.std(ddof=1)
    plt.figure(figsize=(13, 5))
    sns.heatmap(standardized.rename(index=names,columns=SHORT),annot=True,fmt='.2f',cmap='RdBu_r',center=0,
                cbar_kws={'label':'Across-segment standardized median'})
    plt.xticks(rotation=35,ha='right')
    plt.title('Original-unit segment medians, standardized across four segments (ddof=1)')
    save('profile_heatmap')

    p = profiles['proportions']
    fig, axes = plt.subplots(3,1,figsize=(11,9))
    for ax, (field, order) in zip(axes, CATEGORIES.items()):
        subset = p[p.field == field].pivot(index='cluster',columns='category',values='share').reindex(columns=order).fillna(0)
        subset.rename(index=names).plot.barh(stacked=True,ax=ax,colormap='Set2')
        ax.set(title=field.replace('_',' ').title(),xlabel='Within-segment share',ylabel='',xlim=(0,1))
        ax.legend(title='',bbox_to_anchor=(1.01,1),loc='upper left',fontsize=9)
    save('demographic_proportions')

    plt.figure(figsize=(9, 6))
    for cluster in profiles['sizes'].index:
        mask = result['labels'] == cluster
        plt.scatter(result['scores'][mask,0], result['scores'][mask,1],s=12,alpha=.45,color=colors[cluster],label=names[cluster])
    plt.xlabel('PC1 score')
    plt.ylabel('PC2 score')
    plt.title('Two-dimensional projection of the five-PC K-Means solution')
    plt.legend(fontsize=8)
    save('segment_projection')
