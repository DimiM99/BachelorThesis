import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import sys
sys.path.append('../')
from sklmini_py import KMeans, KNeighborsClassifier, LinearRegression
from experiments.benchmark_data import load_real_datasets, Datasets

@dataclass
class BenchmarkResult:
    fit_times: List[float]
    predict_times: List[float]
    metrics: Dict[str, float]

    def __init__(self):
        self.fit_times = []
        self.predict_times = []
        self.metrics = {}

@dataclass
class BenchmarkResults:
    california_lr: BenchmarkResult
    diabetes_lr: BenchmarkResult
    iris_knn: BenchmarkResult
    cancer_knn: BenchmarkResult
    wine_kmeans: BenchmarkResult

    def __init__(self):
        self.california_lr = BenchmarkResult()
        self.diabetes_lr = BenchmarkResult()
        self.iris_knn = BenchmarkResult()
        self.cancer_knn = BenchmarkResult()
        self.wine_kmeans = BenchmarkResult()

def calculate_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    inertia = 0.0
    for i in range(len(centroids)):
        mask = labels == i
        if np.any(mask):
            cluster_points = X[mask]
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
    return inertia

def run_benchmarks(n_runs: int = 5) -> BenchmarkResults:
    results = BenchmarkResults()
    datasets = load_real_datasets()

    # Linear Regression - California Housing
    print("\nBenchmarking Linear Regression on California Housing...")
    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        model = LinearRegression()

        fit_start = time.time()
        model.fit(datasets.california.X_train, datasets.california.y_train_mat)
        fit_time = time.time() - fit_start
        results.california_lr.fit_times.append(fit_time)

        predict_start = time.time()
        y_pred = model.predict(datasets.california.X_test)
        predict_time = time.time() - predict_start
        results.california_lr.predict_times.append(predict_time)

        mse_val = mean_squared_error(datasets.california.y_test_mat, y_pred)
        r2_val = r2_score(datasets.california.y_test_mat, y_pred)
        if i == 0:  # Store metrics from first run
            results.california_lr.metrics["mse"] = mse_val
            results.california_lr.metrics["r2"] = r2_val

    # Linear Regression - Diabetes
    print("\nBenchmarking Linear Regression on Diabetes dataset...")
    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        model = LinearRegression()

        fit_start = time.time()
        model.fit(datasets.diabetes.X_train, datasets.diabetes.y_train_mat)
        fit_time = time.time() - fit_start
        results.diabetes_lr.fit_times.append(fit_time)

        predict_start = time.time()
        y_pred = model.predict(datasets.diabetes.X_test)
        predict_time = time.time() - predict_start
        results.diabetes_lr.predict_times.append(predict_time)

        mse_val = mean_squared_error(datasets.diabetes.y_test_mat, y_pred)
        r2_val = r2_score(datasets.diabetes.y_test_mat, y_pred)
        if i == 0:  # Store metrics from first run
            results.diabetes_lr.metrics["mse"] = mse_val
            results.diabetes_lr.metrics["r2"] = r2_val

    # KNN - Iris
    print("\nBenchmarking KNN on Iris dataset...")
    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        model = KNeighborsClassifier(k=3)

        fit_start = time.time()
        model.fit(datasets.iris.X_train, datasets.iris.y_train)
        fit_time = time.time() - fit_start
        results.iris_knn.fit_times.append(fit_time)

        predict_start = time.time()
        y_pred = model.predict(datasets.iris.X_test)
        predict_time = time.time() - predict_start
        results.iris_knn.predict_times.append(predict_time)

        acc = accuracy_score(datasets.iris.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.iris_knn.metrics["accuracy"] = acc

    # KNN - Breast Cancer
    print("\nBenchmarking KNN on Breast Cancer dataset...")
    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        model = KNeighborsClassifier(k=3)

        fit_start = time.time()
        model.fit(datasets.cancer.X_train, datasets.cancer.y_train)
        fit_time = time.time() - fit_start
        results.cancer_knn.fit_times.append(fit_time)

        predict_start = time.time()
        y_pred = model.predict(datasets.cancer.X_test)
        predict_time = time.time() - predict_start
        results.cancer_knn.predict_times.append(predict_time)

        acc = accuracy_score(datasets.cancer.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.cancer_knn.metrics["accuracy"] = acc

    # KMeans - Wine
    print("\nBenchmarking KMeans on Wine dataset...")
    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        model = KMeans(K=3)  # Wine dataset has 3 classes

        fit_start = time.time()
        labels = model.predict(datasets.wine.X_train)
        fit_time = time.time() - fit_start
        results.wine_kmeans.fit_times.append(fit_time)

        predict_start = time.time()
        test_labels = model.predict(datasets.wine.X_test)
        predict_time = time.time() - predict_start
        results.wine_kmeans.predict_times.append(predict_time)

        inertia = calculate_inertia(datasets.wine.X_train, labels, model.centroids)
        if i == 0:  # Store metrics from first run
            results.wine_kmeans.metrics["inertia"] = inertia

    return results

def print_results(results: BenchmarkResults, n_runs: int):
    print("\n=== Benchmark Results ===")

    print("\nCalifornia Housing - Linear Regression:")
    print(f"Average Fit Time: {np.mean(results.california_lr.fit_times):.6f} seconds")
    print(f"Average Predict Time: {np.mean(results.california_lr.predict_times):.6f} seconds")
    print(f"MSE: {results.california_lr.metrics['mse']:.6f}")
    print(f"R2 Score: {results.california_lr.metrics['r2']:.6f}")

    print("\nDiabetes - Linear Regression:")
    print(f"Average Fit Time: {np.mean(results.diabetes_lr.fit_times):.6f} seconds")
    print(f"Average Predict Time: {np.mean(results.diabetes_lr.predict_times):.6f} seconds")
    print(f"MSE: {results.diabetes_lr.metrics['mse']:.6f}")
    print(f"R2 Score: {results.diabetes_lr.metrics['r2']:.6f}")

    print("\nIris - KNN:")
    print(f"Average Fit Time: {np.mean(results.iris_knn.fit_times):.6f} seconds")
    print(f"Average Predict Time: {np.mean(results.iris_knn.predict_times):.6f} seconds")
    print(f"Accuracy: {results.iris_knn.metrics['accuracy']:.6f}")

    print("\nBreast Cancer - KNN:")
    print(f"Average Fit Time: {np.mean(results.cancer_knn.fit_times):.6f} seconds")
    print(f"Average Predict Time: {np.mean(results.cancer_knn.predict_times):.6f} seconds")
    print(f"Accuracy: {results.cancer_knn.metrics['accuracy']:.6f}")

    print("\nWine - KMeans:")
    print(f"Average Fit Time: {np.mean(results.wine_kmeans.fit_times):.6f} seconds")
    print(f"Average Predict Time: {np.mean(results.wine_kmeans.predict_times):.6f} seconds")
    print(f"Inertia: {results.wine_kmeans.metrics['inertia']:.6f}")

    print("\n=== Timing Variations ===")
    print("\nCalifornia Housing - Linear Regression:")
    print(f"Fit Time Std: {np.std(results.california_lr.fit_times):.6f}")
    print(f"Predict Time Std: {np.std(results.california_lr.predict_times):.6f}")

    print("\nDiabetes - Linear Regression:")
    print(f"Fit Time Std: {np.std(results.diabetes_lr.fit_times):.6f}")
    print(f"Predict Time Std: {np.std(results.diabetes_lr.predict_times):.6f}")

    print("\nIris - KNN:")
    print(f"Fit Time Std: {np.std(results.iris_knn.fit_times):.6f}")
    print(f"Predict Time Std: {np.std(results.iris_knn.predict_times):.6f}")

    print("\nBreast Cancer - KNN:")
    print(f"Fit Time Std: {np.std(results.cancer_knn.fit_times):.6f}")
    print(f"Predict Time Std: {np.std(results.cancer_knn.predict_times):.6f}")

    print("\nWine - KMeans:")
    print(f"Fit Time Std: {np.std(results.wine_kmeans.fit_times):.6f}")
    print(f"Predict Time Std: {np.std(results.wine_kmeans.predict_times):.6f}")
