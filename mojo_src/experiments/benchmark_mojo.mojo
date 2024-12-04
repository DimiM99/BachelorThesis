from time import now
from python import Python, PythonObject
from sklmini_mo.utility.matrix import Matrix
from sklmini_mo.utility.utils import accuracy_score, r2_score, mse
from sklmini_mo.models.KNN import KNN
from sklmini_mo.models.KMeans import KMeans
from sklmini_mo.models.LinearRegression import LinearRegression
from experiments.benchmark_utils import Datasets, load_real_datasets
from dl_custom.cnn import SimpleNN

struct BenchmarkResult:
    var fit_times: PythonObject
    var predict_times: PythonObject
    var metrics: PythonObject

    fn __init__(inout self):
        self.fit_times = Python.list()
        self.predict_times = Python.list()
        self.metrics = Python.dict()

    fn __copyinit__(inout self, other: Self):
        self.fit_times = other.fit_times
        self.predict_times = other.predict_times
        self.metrics = other.metrics

struct BenchmarkResults:
    var california_lr: BenchmarkResult
    var diabetes_lr: BenchmarkResult
    var iris_knn: BenchmarkResult
    var cancer_knn: BenchmarkResult
    var wine_kmeans: BenchmarkResult
    var mnist_simplenn: BenchmarkResult

    fn __init__(inout self):
        self.california_lr = BenchmarkResult()
        self.diabetes_lr = BenchmarkResult()
        self.iris_knn = BenchmarkResult()
        self.cancer_knn = BenchmarkResult()
        self.wine_kmeans = BenchmarkResult()
        self.mnist_simplenn = BenchmarkResult()

    fn __copyinit__(inout self, other: Self):
        self.california_lr = other.california_lr
        self.diabetes_lr = other.diabetes_lr
        self.iris_knn = other.iris_knn
        self.cancer_knn = other.cancer_knn
        self.wine_kmeans = other.wine_kmeans
        self.mnist_simplenn = other.mnist_simplenn

fn calculate_inertia(X: Matrix, labels: Matrix, centroids: Matrix) raises -> Float32:
    var inertia: Float32 = 0.0
    for i in range(centroids.height):
        var cluster_indices = labels.argwhere_l(labels == i)
        if len(cluster_indices) > 0:
            var cluster_points = X[cluster_indices]
            inertia += ((cluster_points - centroids[i]) ** 2).sum()
    return inertia

fn run_benchmarks(n_runs: Int = 5) raises -> BenchmarkResults:
    var results = BenchmarkResults()
    var datasets = load_real_datasets()
    print("Starting benchmarks...")
    
    # Linear Regression - California Housing
    print("\nBenchmarking Linear Regression on California Housing...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = LinearRegression(
            learning_rate=0.01,
            n_iters=1000,
            reg_alpha=0.1,
        )
        
        var fit_start = now()
        model.fit(datasets.california.X_train, datasets.california.y_train)
        var fit_time = (now() - fit_start) / 1e9
        results.california_lr.fit_times.append(fit_time)
        
        var predict_start = now()
        var y_pred = model.predict(datasets.california.X_test)
        var predict_time = (now() - predict_start) / 1e9
        results.california_lr.predict_times.append(predict_time)
        
        var mse_val = mse(datasets.california.y_test, y_pred)
        var r2_val = r2_score(datasets.california.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.california_lr.metrics["mse"] = mse_val
            results.california_lr.metrics["r2"] = r2_val

    # Linear Regression - Diabetes
    print("\nBenchmarking Linear Regression on Diabetes dataset...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = LinearRegression(
            learning_rate=0.01,
            n_iters=1000,
            reg_alpha=0.1,
        )
        
        var fit_start = now()
        model.fit(datasets.diabetes.X_train, datasets.diabetes.y_train)
        var fit_time = (now() - fit_start) / 1e9
        results.diabetes_lr.fit_times.append(fit_time)
        
        var predict_start = now()
        var y_pred = model.predict(datasets.diabetes.X_test)
        var predict_time = (now() - predict_start) / 1e9
        results.diabetes_lr.predict_times.append(predict_time)
        
        var mse_val = mse(datasets.diabetes.y_test, y_pred)
        var r2_val = r2_score(datasets.diabetes.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.diabetes_lr.metrics["mse"] = mse_val
            results.diabetes_lr.metrics["r2"] = r2_val

    # KNN - Iris
    print("\nBenchmarking KNN on Iris dataset...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = KNN(k=3)
        
        var fit_start = now()
        model.fit(datasets.iris.X_train, datasets.iris.y_train)
        var fit_time = (now() - fit_start) / 1e9
        results.iris_knn.fit_times.append(fit_time)
        
        var predict_start = now()
        var y_pred = model.predict(datasets.iris.X_test)
        var predict_time = (now() - predict_start) / 1e9
        results.iris_knn.predict_times.append(predict_time)
        
        var acc = accuracy_score(datasets.iris.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.iris_knn.metrics["accuracy"] = acc

    # KNN - Breast Cancer
    print("\nBenchmarking KNN on Breast Cancer dataset...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = KNN(k=3)
        
        var fit_start = now()
        model.fit(datasets.cancer.X_train, datasets.cancer.y_train)
        var fit_time = (now() - fit_start) / 1e9
        results.cancer_knn.fit_times.append(fit_time)
        
        var predict_start = now()
        var y_pred = model.predict(datasets.cancer.X_test)
        var predict_time = (now() - predict_start) / 1e9
        results.cancer_knn.predict_times.append(predict_time)
        
        var acc = accuracy_score(datasets.cancer.y_test, y_pred)
        if i == 0:  # Store metrics from first run
            results.cancer_knn.metrics["accuracy"] = acc

    # KMeans - Wine
    print("\nBenchmarking KMeans on Wine dataset...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = KMeans(K=3)  # Wine dataset has 3 classes
        
        var fit_start = now()
        var labels = model.predict(datasets.wine.X_train)
        var fit_time = (now() - fit_start) / 1e9
        results.wine_kmeans.fit_times.append(fit_time)
        
        var predict_start = now()
        var test_labels = model.predict(datasets.wine.X_test)
        var predict_time = (now() - predict_start) / 1e9
        results.wine_kmeans.predict_times.append(predict_time)
        
        var inertia = calculate_inertia(datasets.wine.X_train, labels, model.centroids)
        if i == 0:  # Store metrics from first run
            results.wine_kmeans.metrics["inertia"] = inertia
    
    # SimpleNN - MNIST
    print("\nBenchmarking SimpleNN on MNIST dataset...")
    for i in range(n_runs):
        print("Run", i + 1, "of", n_runs)
        var model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        var fit_start = now()
        model.fit(datasets.mnist.X_train, datasets.mnist.y_train, 
                 learning_rate=0.1, epochs=500)
        var fit_time = (now() - fit_start) / 1e9
        results.mnist_simplenn.fit_times.append(fit_time)
        
        var predict_start = now()
        var accuracy = model.evaluate(datasets.mnist.X_test, datasets.mnist.y_test)
        var predict_time = (now() - predict_start) / 1e9
        results.mnist_simplenn.predict_times.append(predict_time)
        
        if i == 0:  # Store metrics from first run
            results.mnist_simplenn.metrics["accuracy"] = accuracy

    return results

fn print_results(results: BenchmarkResults, n_runs: Int) raises:
    var np = Python.import_module("numpy")
    
    print("\n=== Benchmark Results ===")
    
    print("\nCalifornia Housing - Linear Regression:")
    print("Average Fit Time:", np.mean(results.california_lr.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.california_lr.predict_times), "seconds")
    print("MSE:", results.california_lr.metrics["mse"])
    print("R2 Score:", results.california_lr.metrics["r2"])
    
    print("\nDiabetes - Linear Regression:")
    print("Average Fit Time:", np.mean(results.diabetes_lr.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.diabetes_lr.predict_times), "seconds")
    print("MSE:", results.diabetes_lr.metrics["mse"])
    print("R2 Score:", results.diabetes_lr.metrics["r2"])
    
    print("\nIris - KNN:")
    print("Average Fit Time:", np.mean(results.iris_knn.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.iris_knn.predict_times), "seconds")
    print("Accuracy:", results.iris_knn.metrics["accuracy"])
    
    print("\nBreast Cancer - KNN:")
    print("Average Fit Time:", np.mean(results.cancer_knn.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.cancer_knn.predict_times), "seconds")
    print("Accuracy:", results.cancer_knn.metrics["accuracy"])
    
    print("\nWine - KMeans:")
    print("Average Fit Time:", np.mean(results.wine_kmeans.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.wine_kmeans.predict_times), "seconds")
    print("Inertia:", results.wine_kmeans.metrics["inertia"])

    print("\nMNIST - Neural Network:")
    print("Average Fit Time:", np.mean(results.mnist_simplenn.fit_times), "seconds")
    print("Average Predict Time:", np.mean(results.mnist_simplenn.predict_times), "seconds")
    print("Accuracy:", results.mnist_simplenn.metrics["accuracy"])
    
    # Print standard deviations
    print("\n=== Timing Variations ===")
    print("\nCalifornia Housing - Linear Regression:")
    print("Fit Time Std:", np.std(results.california_lr.fit_times))
    print("Predict Time Std:", np.std(results.california_lr.predict_times))
    
    print("\nDiabetes - Linear Regression:")
    print("Fit Time Std:", np.std(results.diabetes_lr.fit_times))
    print("Predict Time Std:", np.std(results.diabetes_lr.predict_times))
    
    print("\nIris - KNN:")
    print("Fit Time Std:", np.std(results.iris_knn.fit_times))
    print("Predict Time Std:", np.std(results.iris_knn.predict_times))
    
    print("\nBreast Cancer - KNN:")
    print("Fit Time Std:", np.std(results.cancer_knn.fit_times))
    print("Predict Time Std:", np.std(results.cancer_knn.predict_times))
    
    print("\nWine - KMeans:")
    print("Fit Time Std:", np.std(results.wine_kmeans.fit_times))
    print("Predict Time Std:", np.std(results.wine_kmeans.predict_times))

    print("\nMNIST - Neural Network:")
    print("Fit Time Std:", np.std(results.mnist_simplenn.fit_times))
    print("Predict Time Std:", np.std(results.mnist_simplenn.predict_times))