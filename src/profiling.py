"""Original-unit aggregate profiles with preserved row identity."""
import numpy as np
import pandas as pd
from .config import NUMERIC, CATEGORIES

def profile_segments(profile, labels):
    labels = np.asarray(labels)
    if len(labels) != len(profile) or profile.customer_id.duplicated().any():
        raise ValueError('Labels must align one-to-one with unique customers in preserved row order.')
    joined = profile.copy()
    joined['cluster'] = labels
    sizes = joined.groupby('cluster').size().rename('count').to_frame()
    sizes['share'] = sizes['count'] / len(joined)
    medians = joined.groupby('cluster')[NUMERIC].median()
    means = joined.groupby('cluster')[NUMERIC].mean()
    modes = joined.groupby('cluster')[list(CATEGORIES)].agg(lambda s: s.mode().iloc[0])
    proportions = []
    for field in CATEGORIES:
        table = joined.groupby(['cluster', field]).size().rename('count').reset_index().rename(columns={field:'category'})
        table['denominator'] = table['cluster'].map(sizes['count'])
        table['share'] = table['count'] / table['denominator']
        table['field'] = field
        proportions.append(table)
    return dict(sizes=sizes, medians=medians, means=means, modes=modes,
                proportions=pd.concat(proportions, ignore_index=True))

def identify_segments(medians):
    """Derive names from observed profiles, independent of cluster ID order."""
    remaining = set(medians.index)
    high = medians.monthly_income.idxmax()
    remaining.remove(high)
    visits = medians[['visits_store_per_month', 'visits_online_per_month']]
    low = (visits / visits.std()).sum(axis=1).loc[sorted(remaining)].idxmin()
    remaining.remove(low)
    family = medians.loc[sorted(remaining), 'spend_grocery'].idxmax()
    remaining.remove(family)
    fashion = remaining.pop()
    return {low:'Low-engagement shoppers', family:'Family-focused shoppers',
            fashion:'Fashion and technology shoppers', high:'High-income omnichannel shoppers'}
