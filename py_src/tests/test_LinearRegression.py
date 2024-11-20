import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
import numpy.testing as npt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklmini_py.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        # Load dataset
        california = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            california.data, california.target, test_size=0.2, random_state=42
        )

        # reshape target
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # SKLmini implementation
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Sklearn implementation
        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        
        # Compare Mean Squared Errors
        mse_my = mean_squared_error(y_test, y_pred)
        mse_sklearn = mean_squared_error(y_test, sklearn_pred)
        
        # Assert that the MSE difference is within an acceptable range
        npt.assert_allclose(mse_my, mse_sklearn, rtol=15)

if __name__ == '__main__':
    unittest.main()