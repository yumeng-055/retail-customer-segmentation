"""Explicit feature order and model settings."""
SEED = 64
FEATURES = ['monthly_income', 'spend_grocery', 'spend_fashion', 'spend_electronics',
            'spend_dining', 'spend_online', 'visits_store_per_month', 'visits_online_per_month']
CATEGORIES = {'gender': ['Male', 'Female'], 'marital_status': ['Single', 'Married', 'Divorced'],
              'education_level': ['High School', 'Bachelor', 'Master', 'PhD']}
NUMERIC = ['age', 'monthly_income', 'num_children', 'spend_grocery', 'spend_fashion',
           'spend_electronics', 'spend_dining', 'spend_online', 'visits_store_per_month',
           'visits_online_per_month', 'loyalty_score']
REQUIRED = ['customer_id', *CATEGORIES, *NUMERIC]

def kmeans_settings(k, n_init=20, seed=SEED):
    """Use explicit settings for every fit."""
    return dict(n_clusters=k, init='k-means++', n_init=n_init, random_state=seed,
                max_iter=300, tol=1e-4, algorithm='lloyd')
