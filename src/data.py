"""Load and validate data without silently changing records."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CATEGORIES, NUMERIC, REQUIRED

def load_data(path):
    if not Path(path).is_file():
        raise FileNotFoundError('Customer CSV unavailable. Restore data/raw/customer_segmentation.csv or pass its path with --data.')
    return pd.read_csv(path)

def validate_data(df):
    """Return aggregate audit tables; fail clearly on unsupported values."""
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if len(df) != 2000:
        raise ValueError(f'Expected 2,000 course customers; received {len(df):,}.')
    if df.customer_id.isna().any() or df.customer_id.duplicated().any() or df.duplicated().any():
        raise ValueError('Missing/duplicate identities or duplicate rows found; no records were removed.')
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f'{col} must be numeric.')
        values = df[col].dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{col} contains nonfinite or negative observed values.')
    for col, allowed in CATEGORIES.items():
        if df[col].isna().any() or not set(df[col]).issubset(allowed):
            raise ValueError(f'Unsupported category values in {col}.')
    allowed_missing = ['monthly_income', 'num_children', 'spend_electronics', 'spend_online']
    if df.drop(columns=allowed_missing).isna().any().any():
        raise ValueError('Missing values outside fields with specified imputation.')
    if not (df.num_children.dropna() % 1 == 0).all():
        raise ValueError('Noninteger child counts found.')
    quality = pd.DataFrame({'field': df.columns, 'dtype': [str(t) for t in df.dtypes],
                            'missing_count': df.isna().sum().to_numpy(),
                            'missing_share': df.isna().mean().to_numpy()})
    checks = pd.DataFrame([
        ('customer_count', len(df)), ('column_count', len(df.columns)),
        ('duplicate_rows', int(df.duplicated().sum())), ('duplicate_ids', int(df.customer_id.duplicated().sum())),
        ('missing_cells', int(df.isna().sum().sum())), ('fields_with_missing_values', int(df.isna().any().sum())),
        ('negative_observed_values', int((df[NUMERIC] < 0).sum().sum())),
        ('age_outside_18_to_100', int(((df.age < 18) | (df.age > 100)).sum())),
        ('income_above_100000_SGD', int((df.monthly_income > 100000).sum())),
        ('store_visits_above_60_per_month', int((df.visits_store_per_month > 60).sum())),
        ('loyalty_outside_0_to_100', int(((df.loyalty_score < 0) | (df.loyalty_score > 100)).sum())),
    ], columns=['check', 'count'])
    ranges = df[NUMERIC].agg(['min', 'max', 'median']).T.rename_axis('field').reset_index()
    categories = pd.concat([df[col].value_counts().rename_axis('category').reset_index(name='count').assign(field=col)
                            for col in CATEGORIES], ignore_index=True)
    categories['share'] = categories['count'] / len(df)
    return dict(data_quality=quality, data_checks=checks, observed_ranges=ranges, category_overview=categories)
