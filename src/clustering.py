"""PCA, candidate evaluation, and aligned stability comparisons."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.utils import resample
from .config import FEATURES, SEED, kmeans_settings

def fit_pca(scaled):
    model = PCA(random_state=SEED, svd_solver='auto', whiten=False)
    scores = model.fit_transform(pd.DataFrame(scaled, columns=FEATURES))
    names = [f'PC{i + 1}' for i in range(len(FEATURES))]
    variance = pd.DataFrame({'PC': names, 'explained_variance': model.explained_variance_ratio_,
                            'cumulative_variance': np.cumsum(model.explained_variance_ratio_)})
    weights = pd.DataFrame(model.components_.T, index=FEATURES, columns=names)
    return model, scores[:, :5], variance, weights

def partition_metrics(x, labels, sample_size=None):
    """Silhouette can be sampled; CH and DB use all observations."""
    return dict(silhouette=silhouette_score(x, labels, sample_size=sample_size, random_state=SEED),
                CHI=calinski_harabasz_score(x, labels), DBI=davies_bouldin_score(x, labels),
                minClusterRatio=pd.Series(labels).value_counts(normalize=True).min())

def evaluate_kmeans(x):
    rows, partitions = [], {}
    for k in range(2, 11):
        model = KMeans(**kmeans_settings(k))
        labels = model.fit_predict(x)
        partitions[k] = labels
        rows.append(dict(K=k, inertia=model.inertia_, **partition_metrics(x, labels, min(1000, len(x)))))
    return pd.DataFrame(rows), partitions

def agreement(a, b):
    return adjusted_rand_score(a, b), normalized_mutual_info_score(a, b)

def summarize_stability(pairs, suffix=''):
    values = np.array([agreement(a, b) for a, b in pairs])
    return {f'{metric}{suffix}_{stat}': float(getattr(values[:, j], stat)())
            for j, metric in enumerate(['ARI', 'NMI']) for stat in ['mean', 'std']}

def kmeans_stability(x):
    """12 subset and 12 single-initialization fits; 11 comparisons each."""
    rows = []
    for k in range(2, 11):
        fits = []
        for seed in range(1000, 1012):
            idx = np.random.RandomState(seed).choice(np.arange(len(x)), size=int(len(x) * .8), replace=False)
            labels = KMeans(**kmeans_settings(k, seed=seed)).fit_predict(x[idx])
            fits.append(pd.Series(labels, index=idx))
        pairs = []
        for run in fits[1:]:
            common = np.intersect1d(fits[0].index, run.index)
            pairs.append((fits[0].loc[common], run.loc[common]))
        sub = summarize_stability(pairs)
        labels = [KMeans(**kmeans_settings(k, n_init=1, seed=seed)).fit_predict(x) for seed in range(2000, 2012)]
        ini = summarize_stability([(labels[0], lab) for lab in labels[1:]], suffix='_init')
        rows.append(dict(K=k, **sub, **ini))
    return pd.DataFrame(rows)

def evaluate_ward(x):
    """Full Euclidean/Ward linkage and silhouette; subset versus full reference."""
    z = linkage(x, method='ward', metric='euclidean')
    coph, _ = cophenet(z, pdist(x, metric='euclidean'))
    partitions = {k: fcluster(z, t=k, criterion='maxclust') - 1 for k in range(2, 9)}
    rows = [dict(K=k, copheneticCorr=round(coph, 3), **partition_metrics(x, lab)) for k, lab in partitions.items()]
    subset_fits = []
    for seed in range(SEED, SEED + 10):
        idx = np.asarray(resample(range(len(x)), n_samples=int(len(x) * .8), replace=False, random_state=seed))
        subset_fits.append((idx, linkage(x[idx], method='ward', metric='euclidean')))
    stability = [dict(K=k, **summarize_stability([(lab[idx], fcluster(tree, t=k, criterion='maxclust') - 1)
                                                for idx, tree in subset_fits])) for k, lab in partitions.items()]
    sklabels = AgglomerativeClustering(n_clusters=4, linkage='ward', metric='euclidean').fit_predict(x)
    return z, pd.DataFrame(rows), pd.DataFrame(stability), partitions, agreement(partitions[4], sklabels)
