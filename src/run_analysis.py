"""Run the complete workflow from the repository root."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from .config import kmeans_settings
from .data import load_data, validate_data
from .preprocessing import prepare_data, distribution_summary
from .clustering import fit_pca, evaluate_kmeans, kmeans_stability, evaluate_ward, partition_metrics, agreement
from .profiling import profile_segments, identify_segments
from .visualization import export_figures

def run_analysis(data_path, output_dir='reports'):
    """Compute all results; export aggregate tables and figures only."""
    raw = load_data(data_path)
    tables = validate_data(raw)
    profile, transformed, scaled, scaler, imputation = prepare_data(raw)
    pca, scores, variance, weights = fit_pca(scaled)
    km_candidates, candidate_labels = evaluate_kmeans(scores)
    km_stability = kmeans_stability(scores)
    model = KMeans(**kmeans_settings(4,n_init=30))
    labels = model.fit_predict(scores)
    k3_model = KMeans(**kmeans_settings(3,n_init=30))
    k3_labels = k3_model.fit_predict(scores)
    z, ward_candidates, ward_stability, ward_labels, ward_equivalence = evaluate_ward(scores)
    profiles = {'k4':profile_segments(profile,labels),'k3':profile_segments(profile,k3_labels),
                'ward':profile_segments(profile,ward_labels[4])}
    names = identify_segments(profiles['k4']['medians'])
    final_rows = [dict(K=k,n_init=30,inertia=fit.inertia_,
                       **partition_metrics(scores,lab,min(1000,len(scores))))
                  for k,fit,lab in [(3,k3_model,k3_labels),(4,model,labels)]]
    final_metrics = pd.DataFrame(final_rows)
    cross_ari, cross_nmi = agreement(labels,ward_labels[4])
    contingency = pd.crosstab(pd.Series(labels,name='KMeans_cluster'),pd.Series(ward_labels[4],name='Ward_cluster'))
    rows, cols = linear_sum_assignment(-contingency.to_numpy())
    ward_match = {int(contingency.columns[c]):int(contingency.index[r]) for r,c in zip(rows,cols)}
    matched = profiles['ward']['medians'].copy()
    matched.index = [names[ward_match[i]] for i in matched.index]
    comparison = pd.DataFrame([
        dict(model='K-Means (n_init=30)', **partition_metrics(scores,labels,1000)),
        dict(model='Ward (SciPy)', **partition_metrics(scores,ward_labels[4])),
    ])
    equivalence = pd.DataFrame([
        dict(comparison='candidate20_vs_final30_K4',ARI=agreement(candidate_labels[4],labels)[0],NMI=agreement(candidate_labels[4],labels)[1]),
        dict(comparison='candidate20_vs_final30_K3',ARI=agreement(candidate_labels[3],k3_labels)[0],NMI=agreement(candidate_labels[3],k3_labels)[1]),
        dict(comparison='SciPy_vs_sklearn_Ward_K4',ARI=ward_equivalence[0],NMI=ward_equivalence[1]),
        dict(comparison='KMeans_vs_Ward_K4',ARI=cross_ari,NMI=cross_nmi),
    ])
    tables.update(imputation=imputation, distribution_before=distribution_summary(profile),
                  distribution_after=distribution_summary(transformed),
                  feature_correlation=transformed.corr(),pca_variance=variance,pca_weights=weights,
                  kmeans_candidates=km_candidates,kmeans_stability=km_stability,
                  final_kmeans_metrics=final_metrics,ward_candidates=ward_candidates,
                  ward_stability=ward_stability,model_comparison=comparison,
                  partition_agreement=equivalence,cross_method_contingency=contingency,
                  k3_k4_contingency=pd.crosstab(pd.Series(k3_labels,name='K3_cluster'),pd.Series(labels,name='K4_cluster')),
                  ward_matched_medians=matched)
    for prefix, items in profiles.items():
        for name, table in items.items(): tables[f'{prefix}_{name}'] = table
    overview = profiles['k4']['sizes'].join(profiles['k4']['medians'])
    overview.insert(0,'segment',[names[i] for i in overview.index])
    tables['segment_overview'] = overview
    result = dict(tables=tables, profiles=profiles,names=names,labels=labels,k3_labels=k3_labels,
                  ward_labels=ward_labels[4],scores=scores,scaled=scaled,pca=pca,scaler=scaler,
                  profile=profile,linkage=z,model=model)
    check_integrity(result,raw)
    output = Path(output_dir)
    (output/'tables').mkdir(parents=True,exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output/'tables'/f'{name}.csv',index=not isinstance(table.index,pd.RangeIndex),float_format='%.12g')
    export_figures(result,output/'figures')
    return result

def check_integrity(result, raw):
    """Check alignment, original units, finite matrices, totals, and proportions."""
    if not np.array_equal(result['profile'].customer_id,raw.customer_id):
        raise AssertionError('Customer row order changed.')
    for col in raw.columns:
        mask = raw[col].notna()
        if not np.array_equal(result['profile'].loc[mask,col],raw.loc[mask,col]):
            raise AssertionError(f'Observed original-unit values changed: {col}')
    assert np.isfinite(result['scaled']).all() and np.isfinite(result['scores']).all()
    for items in result['profiles'].values():
        sizes = items['sizes']
        assert sizes['count'].sum() == len(raw)
        np.testing.assert_allclose(sizes['share'],sizes['count']/len(raw),rtol=0,atol=1e-12)
        assert np.isfinite(items['medians'].to_numpy()).all()
        totals = items['proportions'].groupby(['cluster','field'])['share'].sum()
        np.testing.assert_allclose(totals,1,rtol=0,atol=1e-12)
    assert len(result['names']) == 4 and len(set(result['names'].values())) == 4

def main():
    parser = argparse.ArgumentParser(description='Reproduce the full course-data segmentation analysis.')
    parser.add_argument('--data',default='data/raw/customer_segmentation.csv',help='Path to the customer CSV')
    parser.add_argument('--output',default='reports',help='Aggregate report output directory')
    args = parser.parse_args()
    try:
        result = run_analysis(args.data,args.output)
    except (FileNotFoundError,ValueError) as error:
        parser.exit(2,f'Analysis could not run: {error}\n')
    print(f"Completed analysis for {len(result['labels']):,} customers; aggregate outputs saved.")
    print(result['tables']['segment_overview'][['segment','count','share']].to_string(index=False))

if __name__ == '__main__':
    main()
