"""Focused analytical checks using an authorized local course CSV."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import unittest
import numpy as np
from sklearn.cluster import KMeans
from src.config import FEATURES, kmeans_settings
from src.data import load_data, validate_data
from src.preprocessing import prepare_data
from src.clustering import fit_pca, agreement
from src.profiling import profile_segments, identify_segments

class AnalysisIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('SEGMENTATION_DATA', 'data/raw/customer_segmentation.csv')
        if not path:
            raise unittest.SkipTest('Set SEGMENTATION_DATA to an authorized local course CSV.')
        cls.raw = load_data(path)
        validate_data(cls.raw)
        cls.profile, cls.transformed, cls.scaled, _, _ = prepare_data(cls.raw)
        cls.pca, cls.scores, cls.variance, _ = fit_pca(cls.scaled)
        cls.labels = KMeans(**kmeans_settings(4,n_init=30)).fit_predict(cls.scores)
        cls.profiles = profile_segments(cls.profile,cls.labels)

    def test_schema_and_alignment(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):
            validate_data(self.raw.drop(columns=['spend_online']))
        duplicate = self.raw.copy()
        duplicate.loc[1,'customer_id'] = duplicate.loc[0,'customer_id']
        with self.assertRaisesRegex(ValueError,'identities'):
            validate_data(duplicate)
        np.testing.assert_array_equal(self.profile.customer_id,self.raw.customer_id)
        with self.assertRaisesRegex(ValueError,'align'):
            profile_segments(self.profile,self.labels[:-1])

    def test_original_units_and_finite_model_space(self):
        for field in ['monthly_income','visits_store_per_month']:
            observed = self.raw[field].notna()
            np.testing.assert_array_equal(self.profile.loc[observed,field],self.raw.loc[observed,field])
        self.assertEqual(list(self.transformed.columns),FEATURES)
        self.assertTrue(np.isfinite(self.scaled).all())
        self.assertTrue(np.isfinite(self.scores).all())
        np.testing.assert_allclose(self.variance.cumulative_variance.iloc[4],.815745164813346,atol=1e-10,rtol=0)

    def test_counts_shares_names_and_label_permutations(self):
        sizes = self.profiles['sizes']
        self.assertEqual(sizes['count'].sum(),len(self.raw))
        np.testing.assert_allclose(sizes['share'],sizes['count']/len(self.raw),rtol=0,atol=1e-12)
        totals = self.profiles['proportions'].groupby(['cluster','field'])['share'].sum()
        np.testing.assert_allclose(totals,1,rtol=0,atol=1e-12)
        names = identify_segments(self.profiles['medians'])
        expected = {'Low-engagement shoppers':414,'Family-focused shoppers':512,
                    'Fashion and technology shoppers':744,'High-income omnichannel shoppers':330}
        self.assertEqual({names[k]:int(n) for k,n in sizes['count'].items()},expected)
        permuted = profile_segments(self.profile,3-self.labels)
        new_names = identify_segments(permuted['medians'])
        self.assertEqual({new_names[k]:int(n) for k,n in permuted['sizes']['count'].items()},expected)
        self.assertEqual(agreement(self.labels,3-self.labels),(1.,1.))

if __name__ == '__main__':
    unittest.main()
