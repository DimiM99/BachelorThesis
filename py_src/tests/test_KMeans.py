import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklmini_py.kmeans import KMeans

class TestKMeans(unittest.TestCase):
    def test_kmeans_clustering(self):
        # Load dataset
        X, y = load_wine(return_X_y=True)
        X = X.astype(np.float32)
        
        # Our implementation
        model = KMeans(K=3, max_iters=100)
        labels_our = model.predict(X)
        
        # Sklearn implementation
        sklearn_model = SklearnKMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=42)
        sklearn_model.fit(X)
        labels_sklearn = sklearn_model.labels_
        
        # Compare clustering results using Adjusted Rand Index
        ari_our = adjusted_rand_score(y, labels_our)
        ari_sklearn = adjusted_rand_score(y, labels_sklearn)
        
        # Assert that the ARI scores are within an acceptable range
        self.assertAlmostEqual(ari_our, ari_sklearn, delta=0.1)

if __name__ == '__main__':
    unittest.main()