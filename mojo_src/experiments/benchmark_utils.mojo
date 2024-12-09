from python import Python, PythonObject
from sklmini_mo.utility.matrix import Matrix
from memory import memcpy

struct DatasetResult:
    var X_train: Matrix
    var X_test: Matrix
    var y_train: Matrix
    var y_test: Matrix

    fn __init__(inout self):
        self.X_train = Matrix(0, 0)
        self.X_test = Matrix(0, 0)
        self.y_train = Matrix(0, 0)
        self.y_test = Matrix(0, 0)

    fn __copyinit__(inout self, other: Self):
        self.X_train = other.X_train
        self.X_test = other.X_test
        self.y_train = other.y_train
        self.y_test = other.y_test
        

struct Datasets:
    var iris: DatasetResult
    var wine: DatasetResult
    var california: DatasetResult
    var diabetes: DatasetResult
    var cancer: DatasetResult
    var mnist: DatasetResult 

    fn __init__(inout self):
        self.iris = DatasetResult()
        self.wine = DatasetResult()
        self.california = DatasetResult()
        self.diabetes = DatasetResult()
        self.cancer = DatasetResult()
        self.mnist = DatasetResult()

    fn __copyinit__(inout self, other: Self):
        self.iris = other.iris
        self.wine = other.wine
        self.california = other.california
        self.diabetes = other.diabetes
        self.cancer = other.cancer
        self.mnist = other.mnist

fn load_real_datasets() raises -> Datasets:
    print("Loading the datasets...")
    var datasets = Datasets()
    
    var sklearn_datasets = Python.import_module("sklearn.datasets")
    var pd = Python.import_module("pandas")
    var np = Python.import_module("numpy")
    var sklearn_model_selection = Python.import_module("sklearn.model_selection")
    
    # California Housing dataset (regression)
    print("Loading California Housing dataset...")
    var california = sklearn_datasets.fetch_california_housing()
    var california_split = sklearn_model_selection.train_test_split(
        california.data, california.target, test_size=0.2, random_state=42
    )
    datasets.california.X_train = Matrix.from_numpy(california_split[0])
    datasets.california.X_test = Matrix.from_numpy(california_split[1])
    datasets.california.y_train = Matrix.from_numpy(california_split[2].reshape(-1, 1))
    datasets.california.y_test = Matrix.from_numpy(california_split[3].reshape(-1, 1))

    # Diabetes dataset (regression)
    print("Loading Diabetes dataset...")
    var diabetes = sklearn_datasets.load_diabetes()
    var diabetes_split = sklearn_model_selection.train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )
    datasets.diabetes.X_train = Matrix.from_numpy(diabetes_split[0])
    datasets.diabetes.X_test = Matrix.from_numpy(diabetes_split[1])
    datasets.diabetes.y_train = Matrix.from_numpy(diabetes_split[2].reshape(-1, 1))
    datasets.diabetes.y_test = Matrix.from_numpy(diabetes_split[3].reshape(-1, 1))

    # Iris dataset (classification)
    print("Loading Iris dataset...")
    var iris = sklearn_datasets.load_iris()
    var iris_split = sklearn_model_selection.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    datasets.iris.X_train = Matrix.from_numpy(iris_split[0])
    datasets.iris.X_test = Matrix.from_numpy(iris_split[1])
    datasets.iris.y_train = Matrix.from_numpy(iris_split[2].reshape(-1, 1))
    datasets.iris.y_test = Matrix.from_numpy(iris_split[3].reshape(-1, 1))

    # Wine dataset (clustering/classification)
    print("Loading Wine dataset...")
    var wine = sklearn_datasets.load_wine()
    var wine_split = sklearn_model_selection.train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    datasets.wine.X_train = Matrix.from_numpy(wine_split[0])
    datasets.wine.X_test = Matrix.from_numpy(wine_split[1])
    datasets.wine.y_train = Matrix.from_numpy(wine_split[2].reshape(-1, 1))
    datasets.wine.y_test = Matrix.from_numpy(wine_split[3].reshape(-1, 1))
    
    # Breast Cancer dataset (classification)
    print("Loading Breast Cancer dataset...")
    var cancer = sklearn_datasets.load_breast_cancer()
    var cancer_split = sklearn_model_selection.train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    datasets.cancer.X_train = Matrix.from_numpy(cancer_split[0])
    datasets.cancer.X_test = Matrix.from_numpy(cancer_split[1])
    datasets.cancer.y_train = Matrix.from_numpy(cancer_split[2].reshape(-1, 1))
    datasets.cancer.y_test = Matrix.from_numpy(cancer_split[3].reshape(-1, 1))

    # MNIST dataset (classification)
    print("Loading MNIST Digits Dataset...")
    print("Checking if dataset is already downloaded...")
    var data: PythonObject = pd.DataFrame()
    try:
        data = pd.read_csv("mnist.csv", index_col=False)
        print("Dataset loaded from a local copy.")
    except:
        print("Dataset is not found. Downloading from the GCS Bucket... (may take a while)")
        data = pd.read_csv("https://storage.googleapis.com/mnist-test-mojo-ba/mnist.csv", index_col=False)
        data.to_csv("mnist.csv", index = False)
        print("Dataset downloaded and saved to the local directory.")
    np_data = np.array(data)
    var mnist = Matrix.from_numpy(np_data)
    var m = mnist.height
    var n = mnist.width
    var split_idx = 1000

    data_test = mnist[0:split_idx, :].T()
    Y_test = data_test[0]
    X_test = data_test[1:n, :]
    X_test = X_test / 255

    data_train = mnist[split_idx:m, :].T()
    Y_train = data_train[0]
    X_train = data_train[1:n, :]
    X_train = X_train / 255

    datasets.mnist.X_train = X_train
    datasets.mnist.X_test = X_test
    datasets.mnist.y_train = Y_train
    datasets.mnist.y_test = Y_test

    
    print("Datasets loaded.")
    return datasets