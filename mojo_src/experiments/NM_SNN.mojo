from builtin.file import open
from numojo.prelude import *
import numojo as nm

from algorithm import vectorize, parallelize
from sys.info import simdwidthof, num_logical_cores

from algorithm.reduction import sum, max
from buffer import Buffer, NDBuffer, DimList

from time.time import perf_counter_ns


struct MN_SimpleNN:
    var input_size: Int
    var hidden_size: Int
    var output_size: Int
    var num_of_observations: Int

    var debug: Bool

    var W1: NDArray[DType.float64]
    var b1: NDArray[DType.float64]
    var W2: NDArray[DType.float64]
    var b2: NDArray[DType.float64]

    var Z1: NDArray[DType.float64]
    var A1: NDArray[DType.float64]
    var Z2: NDArray[DType.float64]
    var A2: NDArray[DType.float64]

    fn __init__(
        mut self,
        input_size: Int = 784,
        hidden_size: Int = 10,
        output_size: Int = 10,
        num_of_observations: Int = 42000,
        debug: Bool = False
    ) raises :
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_of_observations = num_of_observations
        self.debug = debug

        # Initialize weights and biases
        self.W1 = nm.random.randn(self.hidden_size, self.input_size)
        self.b1 = nm.random.randn(self.hidden_size, 1)
        self.W2 = nm.random.randn(self.output_size, self.hidden_size)
        self.b2 = nm.random.randn(self.output_size, 1)

        # Initialize activations and linear combinations
        self.Z1 = nm.random.randn(self.hidden_size, self.num_of_observations)
        self.A1 = nm.random.randn(self.hidden_size, self.num_of_observations)
        self.Z2 = nm.random.randn(self.output_size, self.num_of_observations)
        self.A2 = nm.random.randn(self.output_size, self.num_of_observations)

    @staticmethod
    fn relu(Z: NDArray[DType.float64]) -> NDArray[DType.float64]:
        var result = Z
        maximum(result, 0.0)
        return result^

    @staticmethod
    fn relu_deriv(Z: NDArray[DType.float64]) -> NDArray[DType.float64]:
        var result = Z
        @parameter
        fn deriv_parallel(i: Int):
            result._buf[i] = 1.0 if Z._buf[i] > 0.0 else 0.0
        parallelize[deriv_parallel](Z.num_elements(), num_logical_cores())
        return result^
        
    @staticmethod
    fn softmax(Z: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
        var exp_Z = nm.math.exp(Z)
        var sum_exp_Z = nm.math.sum(exp_Z)
        return exp_Z / sum_exp_Z

    @staticmethod
    fn one_hot(Y: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
        var num_classes = (_max(Y) + 1).cast[DType.int32]().value
        var num_samples = Y.num_elements()
        var result = nm.creation.zeros(Shape(num_samples, num_classes))
        @parameter
        fn one_hot_parallel(i: Int):
            try: 
                result.itemset(List(i, (Y[Idx(0, i)]).cast[DType.int32]().value), 1.0)
            except:
                pass
        parallelize[one_hot_parallel](num_samples, num_logical_cores())
        return result

    fn forward(
        mut self,
        X: NDArray[DType.float64]
    ) raises -> NDArray[DType.float64]:
        self.Z1 = nm.math.add(self.W1.mdot(X), self.b1)
        self.A1 = Self.relu(self.Z1)
        self.Z2 = nm.math.add(self.W2.mdot(self.A1), self.b2)
        self.A2 = Self.softmax(self.Z2)
        return self.A2

    fn backward(
        mut self,
        X: NDArray[DType.float64],
        Y: NDArray[DType.float64],
        m: Int
    ) raises -> (NDArray[DType.float64], NDArray[DType.float64], NDArray[DType.float64], NDArray[DType.float64]):
        var one_hot_Y = self.one_hot(Y)
        var dZ2 = self.A2 - one_hot_Y
        var dW2 = 1.0 / m * dZ2.mdot(self.A1.T())
        var db2 = 1.0 / m * nm.math.sum(dZ2, 1)
        var dZ1 = self.W2.T().mdot(dZ2) * self.relu_deriv(self.Z1)
        var dW1 = 1.0 / m * dZ1.mdot(X.T())
        var db1 = 1.0 / m * nm.math.sum(dZ1, 1)
        return dW1, db1, dW2, db2

    fn update_params(
        mut self, 
        dW1: NDArray[DType.float64], 
        db1: NDArray[DType.float64], 
        dW2: NDArray[DType.float64], 
        db2: NDArray[DType.float64], 
        learning_rate: Float64
    ) raises:
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    @staticmethod
    fn get_predictions(A2: NDArray[DType.float64]) raises -> NDArray[DType.float64]:
        var m = A2.shape[1]
        var predictions = nm.creation.zeros(Shape(m))
        for i in range(A2.shape[1]):
            var col = A2[:, ]
            predictions[Idx(i)] = nm.math.argmax(col)
        return predictions

    @staticmethod
    fn get_accuracy(predictions: NDArray[DType.float64], Y: NDArray[DType.float64]) raises -> Float64:
        var correct = 0
        for i in range(predictions.shape[0]):
            if predictions[i] == Y[i]:
                correct += 1
        return correct / predictions.shape[0]

    fn fit(
        mut self, 
        X_train: NDArray[DType.float64], 
        Y_train: NDArray[DType.float64], 
        learning_rate: Float64 = 0.1, 
        epochs: Int = 100
    ) raises:
        for epoch in range(epochs):
            # Forward pass
            var A2 = self.forward(X_train)
            
            # Backward pass
            var bp = self.backward(X_train, Y_train, X_train.shape[1])
            dW1, db1, dW2, db2 = bp

            # Update parameters
            self.update_params(dW1, db1, dW2, db2, learning_rate)
            
            # Optional: print debug information
            if self.debug and epoch % 10 == 0:
                var predictions = Self.get_predictions(A2)
                var accuracy = Self.get_accuracy(predictions, Y_train)
                print("Epoch: ", epoch, " Accuracy: ", accuracy)

    fn predict(
        mut self, 
        X: NDArray[DType.float64]
    ) raises -> NDArray[DType.float64]:
        var A2 = self.forward(X)
        return Self.get_predictions(A2)

    fn evaluate(
        mut self, 
        X: NDArray[DType.float64], 
        Y: NDArray[DType.float64]
    ) raises -> Float64:
        var predictions = self.predict(X)
        return Self.get_accuracy(predictions, Y)



fn maximum[type: DType = DType.float64](mut arr: NDArray[type], val: SIMD[type, 1]):
    @parameter
    fn maximum(idx: Int):
        if arr._buf[idx] < val:
            arr._buf[idx] = val
    parallelize[maximum](arr.num_elements(), num_logical_cores())  


fn _max(z: NDArray[DType.float64]) -> Float64:
    try:
        return max(
            Buffer[DType.float64](
                    z.unsafe_ptr(), 
                    z.num_elements()
                )
            )
    except:
        return 0.0  


fn read_file_content(file_path: String) raises -> String:
    var file = open(file_path, "r")
    var content = file.read()
    file.close()
    return content

fn parse_csv(file_path: String, delimiter: String = ",", has_header: Bool = False) raises -> NDArray[DType.float64]:
    # Read the entire file content
    var file_content = read_file_content(file_path)
    
    # Split the content into lines
    var lines = file_content.split("\n")
    
    # Determine start index based on header
    var start_index = 0 if not has_header else 1
    
    # Count rows and columns
    var rows = len(lines) - start_index
    var cols = len(lines[start_index].split(delimiter))
    
    # Create NDArray
    var result = nm.random.randn(rows, cols)
    
    # Parse lines into NDArray
    for i in range(start_index, len(lines)):
        var row = lines[i].split(delimiter)
        for j in range(min(cols, len(row))):
            # Safely convert to float, handling potential parsing errors
            try:
                result[Idx(i - start_index, j)] = atof(row[j].strip())
            except:
                # Optional: handle parsing errors (e.g., set to 0 or skip)
                result[Idx(i - start_index, j)] = 0.0
    
    return result

fn prepare_data(data_path: String) raises -> (NDArray[DType.float64], NDArray[DType.float64], NDArray[DType.float64], NDArray[DType.float64]):
    print("Loading MNIST Digits Dataset...")
    print("Checking if dataset is already downloaded...")
    
    var data: NDArray[DType.float64]
    try:
        data = parse_csv("mnist.csv")
        print("Dataset loaded from a local copy.")
    except:
        print("Dataset is not found. Downloading from the GCS Bucket... (may take a while)")
        data = parse_csv("https://storage.googleapis.com/mnist-test-mojo-ba/mnist.csv")
        # In Python, this used to_csv, but we'll skip that for now
    
    var m = data.shape[0]
    var n = data.shape[1]

    # Split into train and validation
    var split_idx = 1000

    # Prepare validation data
    var data_dev = data[0:split_idx, :].T()
    var Y_dev = data_dev[0, ]
    var X_dev = data_dev[1:n, :]
    X_dev = X_dev / 255.0

    # Prepare training data
    var data_train = data[split_idx:m, :].T()
    var Y_train = data_train[0, ]
    var X_train = data_train[1:n, :]
    X_train = X_train / 255.0

    return X_train, Y_train, X_dev, Y_dev

fn run_test() raises:
    # Load and prepare data
    print("Preparing data...")
    X_train, Y_train, X_val, Y_val = prepare_data('https://storage.googleapis.com/mnist-test-mojo-ba/mnist.csv')
    
    # Create and train model
    print("Training model...")
    var model = MN_SimpleNN(input_size=784, hidden_size=10, output_size=10, debug=True)
    var start_time = perf_counter_ns()
    model.fit(X_train, Y_train, learning_rate=0.1, epochs=500)
    var end_time = perf_counter_ns()

    # Evaluate on validation set
    var val_accuracy = model.evaluate(X_val, Y_val)
    print("Validation accuracy: ", val_accuracy)
    print("Training time: ", (end_time - start_time) / 1e9, " seconds")
