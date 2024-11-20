import numpy as np
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                            load_diabetes, fetch_california_housing)
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class DatasetResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: Any  # Can be either ndarray or array-like for different algorithms
    y_test: Any
    y_train_mat: Optional[np.ndarray] = None  # For Linear Regression
    y_test_mat: Optional[np.ndarray] = None   # For Linear Regression

@dataclass
class Datasets:
    california: DatasetResult
    iris: DatasetResult
    wine: DatasetResult
    diabetes: DatasetResult
    cancer: DatasetResult

def load_real_datasets() -> Datasets:
    # California Housing dataset (regression)
    print("Loading California Housing dataset...")
    california = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        california.data, california.target, test_size=0.2, random_state=42
    )
    california_result = DatasetResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_mat=y_train.reshape(-1, 1),
        y_test_mat=y_test.reshape(-1, 1)
    )

    # Diabetes dataset (regression)
    print("Loading Diabetes dataset...")
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )
    diabetes_result = DatasetResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_mat=y_train.reshape(-1, 1),
        y_test_mat=y_test.reshape(-1, 1)
    )

    # Iris dataset (classification)
    print("Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    iris_result = DatasetResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # Wine dataset (clustering/classification)
    print("Loading Wine dataset...")
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    wine_result = DatasetResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # Breast Cancer dataset (classification)
    print("Loading Breast Cancer dataset...")
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    cancer_result = DatasetResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    return Datasets(
        california=california_result,
        iris=iris_result,
        wine=wine_result,
        diabetes=diabetes_result,
        cancer=cancer_result
    )
