import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklmini_py.knn_classifier import KNeighborsClassifier as KNN

class TestKNN(unittest.TestCase):
    def test_knn_classifier(self):
        # Load dataset
        X, y = load_iris(return_X_y=True)
        X = X.astype(np.float32)
        
        # Split into training and testing sets
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # SKLmini implementation
        model = KNN(k=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Sklearn implementation
        sklearn_model = KNeighborsClassifier(n_neighbors=3)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        
        # Compare Accuracies
        acc_my = accuracy_score(y_test, y_pred)
        acc_sklearn = accuracy_score(y_test, sklearn_pred)
        
        # Assert that the accuracies are within an acceptable range
        self.assertAlmostEqual(acc_my, acc_sklearn, delta=0.1)

if __name__ == '__main__':
    unittest.main()