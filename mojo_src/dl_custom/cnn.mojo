from sklmini_mo.utility.utils import Matrix, ReLu, ReLu_Deriv, softmax, sigmoid, accuracy_score
import math
from time import sleep

struct SimpleNN:
    var input_size: Int
    var hidden_size: Int 
    var output_size: Int
    var debug: Bool
    var debug1: Bool
    var W1: Matrix
    var b1: Matrix 
    var W2: Matrix
    var b2: Matrix
    
    # Model state during forward pass
    var Z1: Matrix
    var A1: Matrix
    var Z2: Matrix 
    var A2: Matrix

    fn __init__(inout self, input_size: Int = 784, hidden_size: Int = 10, output_size: Int = 10, debug: Bool = False, debug1: Bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.debug = debug
        self.debug1 = debug1
        
        # Initialize with random weights - 0.5
        self.W1 = Matrix.rand(self.hidden_size, self.input_size) - 0.5
        self.b1 = Matrix.rand(self.hidden_size, 1) - 0.5
        self.W2 = Matrix.rand(self.output_size, self.hidden_size) - 0.5
        self.b2 = Matrix.rand(self.output_size, 1) - 0.5
        
        # Initialize intermediate state matrices
        self.Z1 = Matrix(0,0)
        self.A1 = Matrix(0,0)
        self.Z2 = Matrix(0,0)
        self.A2 = Matrix(0,0)

    fn one_hot(self, Y: Matrix) raises -> Matrix:
        var one_hot_Y = Matrix.zeros(Y.size, Y.max().cast[DType.int32]().value + 1)
        for i in range(Y.size):
            one_hot_Y[i, Y[0, i].cast[DType.int32]().value] = 1.0
        return one_hot_Y.T()

    fn forward(inout self, X: Matrix) raises -> Matrix:
        self.Z1 = self.W1 * X + self.b1 
        self.A1 = ReLu(self.Z1)
        self.Z2 = self.W2 * self.A1 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    fn backward(self, X: Matrix, Y: Matrix, m: Int) raises -> (Matrix, Matrix, Matrix, Matrix):
        var one_hot_Y = self.one_hot(Y)
        var dZ2 = self.A2 - one_hot_Y
        var dW2 = dZ2 * self.A1.T() / m
        var db2 = dZ2.sum(1).reshape(-1, 1) / m
        var dZ1 = (self.W2.T() * dZ2).ele_mul(ReLu_Deriv(self.Z1))
        var dW1 = dZ1 * X.T() / m
        var db1 = dZ1.sum(1).reshape(-1, 1) / m
        return dW1, db1, dW2, db2

    fn update_params(inout self, dW1: Matrix, db1: Matrix, dW2: Matrix, db2: Matrix, learning_rate: Float64) raises:
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    fn get_predictions(self, A2: Matrix) raises -> Matrix:
        return A2.argmax(0)

    fn get_accuracy(self, predictions: Matrix, Y: Matrix) raises -> Float64:
        return accuracy_score(Y, predictions)

    fn fit(inout self, X_train: Matrix, Y_train: Matrix, learning_rate: Float64 = 0.1, epochs: Int = 100) raises:
        var m = X_train.width
        
        for epoch in range(epochs):
            if epoch == 16 or epoch == 17: # enable debugging for epochs
                self.debug1 = True
                print("DEBUGGING on epoch", epoch, " started")
                print("=====================================")

            # Forward propagation
            var A2 = self.forward(X_train)
            
            # Backward propagation
            var bp = self.backward(X_train, Y_train, m)
            dW1, db1, dW2, db2 = bp
            
            # Update parameters
            self.update_params(dW1, db1, dW2, db2, learning_rate)

            if self.debug and epoch % 10 == 0:
                var predictions = self.get_predictions(A2)
                var accuracy = self.get_accuracy(predictions, Y_train)
                print("Epoch:", epoch, "Accuracy:", accuracy)


            if self.debug1:
                print("DEBUGGING on epoch", epoch, " ended")
                print("=====================================")
                self.debug1 = False

    fn predict(inout self, X: Matrix) raises -> Matrix:
        var A2 = self.forward(X)
        return self.get_predictions(A2)

    fn evaluate(inout self, X: Matrix, Y: Matrix) raises -> Float64:
        var predictions = self.predict(X)
        return self.get_accuracy(predictions, Y)