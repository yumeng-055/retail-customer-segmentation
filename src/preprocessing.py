"""Separate original-unit profiles from transformed model features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import FEATURES

def prepare_data(df):
    """Impute, copy, log1p income/store visits, then standardize."""
    profile = df.copy(deep=True)
    settings = []
    for col, method in [('monthly_income', 'median'), ('num_children', 'median'),
                        ('spend_online', 'mean'), ('spend_electronics', 'mean')]:
        value = getattr(profile[col], method)()
        profile[col] = profile[col].fillna(value)
        settings.append(dict(field=col, imputation=method, fill_value=value))
    profile['num_children'] = profile['num_children'].astype(int)
    transformed = profile[FEATURES].copy(deep=True)
    transformed[['monthly_income', 'visits_store_per_month']] = np.log1p(
        transformed[['monthly_income', 'visits_store_per_month']])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    if not np.isfinite(scaled).all():
        raise ValueError('Nonfinite transformed features.')
    return profile, transformed, scaled, scaler, pd.DataFrame(settings)

def distribution_summary(frame):
    """Skewness and fraction outside 1.5-IQR fences; no removal."""
    values = frame[FEATURES]
    q1, q3 = values.quantile(.25), values.quantile(.75)
    flags = (values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))
    return pd.DataFrame({'skewness': values.skew(), 'outside_IQR_share': flags.mean()})
