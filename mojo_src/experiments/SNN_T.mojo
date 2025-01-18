from tensor import Tensor, TensorShape
from math import exp, ceil
from memory import memset_zero
from utils.index import Index
from algorithm import vectorize, parallelize
from collections import InlinedFixedVector, Dict, Optional
from sys.info import simdwidthof, num_logical_cores
from pathlib import Path
from time import perf_counter_ns
from algorithm.reduction import sum, max
from buffer import Buffer, NDBuffer, DimList

struct SimpleNN:
    var input_size: Int
    var hidden_size: Int
    var output_size: Int
    var num_of_observations: Int

    var debug: Bool

    var W1: Tensor[DType.float64]
    var b1: Tensor[DType.float64]
    var W2: Tensor[DType.float64]
    var b2: Tensor[DType.float64]

    var Z1: Tensor[DType.float64]
    var A1: Tensor[DType.float64]
    var Z2: Tensor[DType.float64]
    var A2: Tensor[DType.float64]
    
    fn __init__(
        mut self, 
        input_size: Int = 784, 
        hidden_size: Int = 10, 
        output_size: Int = 10, 
        debug: Bool = False,
        num_of_observations: Int = 42000
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_of_observations = num_of_observations
        
        self.debug = debug

        self.W1 = Tensor[DType.float64].rand(TensorShape(self.hidden_size, self.input_size)) - 0.5
        self.b1 = Tensor[DType.float64].rand(TensorShape(self.hidden_size, 1)) - 0.5
        self.W2 = Tensor[DType.float64].rand(TensorShape(self.output_size, self.hidden_size)) - 0.5
        self.b2 = Tensor[DType.float64].rand(TensorShape(self.output_size, 1)) - 0.5

        self.Z1 = Tensor[DType.float64](TensorShape(self.hidden_size, self.num_of_observations))
        self.A1 = Tensor[DType.float64](TensorShape(self.hidden_size, self.num_of_observations))
        self.Z2 = Tensor[DType.float64](TensorShape(self.output_size, self.num_of_observations))
        self.A2 = Tensor[DType.float64](TensorShape(self.output_size, self.num_of_observations))

    @staticmethod
    fn relu(Z: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var result = Z
        @parameter
        fn par_relu(idx: Int):
            if result.load(idx) < 0.0:
                result.store(idx, 0.0)
        parallelize[par_relu](Z.num_elements(), num_logical_cores())
        return result

    @staticmethod
    fn relu_deriv(Z: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var result = Tensor[DType.float64](Z.shape())
        if Z.num_elements() < 768:
            for i in range(Z.num_elements()):
                result[i] = 1.0 if Z[i] > 0.0 else 0.0
        else:
            @parameter
            fn deriv_parallel(i: Int):
                result[i] = 1.0 if Z[i] > 0.0 else 0.0
            parallelize[deriv_parallel](Z.num_elements(), num_logical_cores())
        
        return result^

    @staticmethod
    fn softmax(Z: Tensor[DType.float64]) raises -> Tensor[DType.float64]:
        var exp_Z = Self._exp_tensor(Z)
        var sums = Self._sum_tensor(exp_Z, axis=0, keepdim=True)
        var shape = Z.shape()
        var result = Tensor[DType.float64](shape)
        if shape[1] < 768:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    result[i * shape[1] + j] = exp_Z[i * shape[1] + j] / sums[j]
        else:
            @parameter
            fn normalize(idx: Int):
                var i = idx // shape[1]
                var j = idx % shape[1]
                result[i * shape[1] + j] = exp_Z[i * shape[1] + j] / sums[j] 
            parallelize[normalize](Z.num_elements(), num_logical_cores())
        return result^

    @staticmethod
    fn one_hot(Y: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var num_classes = (Self._max(Y) + 1).cast[DType.int32]().value
        var num_samples = Y.num_elements()
        var result = Tensor[DType.float64](TensorShape(num_samples, num_classes))
        memset_zero(result.unsafe_ptr(), result.num_elements())
        if num_samples < 768:
            for i in range(num_samples):
                result[VariadicList(i, (Y[i]).cast[DType.int32]().value)] = 1.0
        else:
            @parameter
            fn one_hot_parallel(i: Int):
                result[VariadicList(i, (Y[i]).cast[DType.int32]().value)] = 1.0
            parallelize[one_hot_parallel](num_samples, num_logical_cores())
        return result^


    fn forward(
        mut self, 
        X: Tensor[DType.float64]
    ) raises -> Tensor[DType.float64]:
        self.Z1 = Self._matmul(self.W1, X)
        self.add_broadcast(self.Z1, self.b1)
        self.A1 = Self.relu(self.Z1)
        self.Z2 = Self._matmul(self.W2, self.A1)
        self.add_broadcast(self.Z2, self.b2)
        self.A2 = Self.softmax(self.Z2)
        return self.A2

    fn backward(
        self, 
        X: Tensor[DType.float64], 
        Y: Tensor[DType.float64],
        m: Int
    ) raises -> (Tensor[DType.float64], Tensor[DType.float64], Tensor[DType.float64], Tensor[DType.float64]):
        var one_hot_Y = Self._transpose_tensor(Self.one_hot(Y)) 
        var dZ2 = self.A2 - one_hot_Y
        var dW2 = 1 / m * Self._matmul(dZ2, Self._transpose_tensor(self.A1))
        var db2 = 1 / m * Self._sum_tensor(dZ2, axis = 1, keepdim = True)
        var dZ1 = Self._matmul(Self._transpose_tensor(self.W2), dZ2) * Self.relu_deriv(self.Z1)
        var dW1 = 1 / m * Self._matmul(dZ1, Self._transpose_tensor(X))
        var db1 = 1/ m * Self._sum_tensor(dZ1, axis = 1, keepdim = True)
        return dW1, db1, dW2, db2


    fn update_params(
        mut self, 
        dW1: Tensor[DType.float64], 
        db1: Tensor[DType.float64],
        dW2: Tensor[DType.float64], 
        db2: Tensor[DType.float64], 
        learning_rate: Float64
    ) raises:
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    fn get_predictions(
        self, 
        A2: Tensor[DType.float64]
    ) -> Tensor[DType.float64]:
        var shape = A2.shape()
        var predictions = Tensor[DType.float64](TensorShape(shape[1]))
        if shape[1] < 768:
            for i in range(shape[1]):
                var max_val = A2[0]
                var max_idx: Float64 = 0
                for j in range(shape[0]):
                    if A2[j] > max_val:
                        max_val = A2[j]
                        max_idx = Float64(j)
                predictions[i] = max_idx
        else:
            @parameter
            fn find_max_parallel(i: Int):
                var max_val = A2[0]
                var max_idx: Float64 = 0
                for j in range(shape[0]):
                    if A2[j] > max_val:
                        max_val = A2[j]
                        max_idx = Float64(j)
                predictions[i] = max_idx
            parallelize[find_max_parallel](shape[1], num_logical_cores())
        return predictions^

    fn get_accuracy(
        self, 
        predictions: Tensor[DType.float64], 
        Y: Tensor[DType.float64]
    ) -> Float64:
        var correct: Float64 = 0.0
        var total = predictions.num_elements()
        
        if predictions.num_elements() < 768:
            for i in range(predictions.num_elements()):
                if predictions[i] == Y[i]:
                    correct += 1
        else:
            @parameter
            fn count_correct_parallel(i: Int):
                if predictions[i] == Y[i]:
                    correct += 1
            parallelize[count_correct_parallel](predictions.num_elements(), num_logical_cores())
        
        return correct / total
        
    fn fit(
        mut self, 
        mut X_train: Tensor[DType.float64], 
        Y_train: Tensor[DType.float64],
        learning_rate: Float64 = 0.1, 
        epochs: Int = 100
    ) raises:
        var m = X_train.shape()[1]
        
        for epoch in range(epochs):
            # Forward propagation
            var A2 = self.forward(X_train)
            # Backward propagation
            var bp_r = self.backward(X_train, Y_train, m)
            dW1, db1, dW2, db2 = bp_r
            # Update parameters
            self.update_params(dW1, db1, dW2, db2, learning_rate)
            if self.debug and epoch % 10 == 0:
                var predictions = self.get_predictions(A2)
                var accuracy = self.get_accuracy(predictions, Y_train)
                print("Epoch:", epoch, "Accuracy:", accuracy)

    fn predict(
        mut self,
        X: Tensor[DType.float64]
    ) raises -> Tensor[DType.float64]:
        var A2 = self.forward(X)
        return self.get_predictions(A2)

    fn evaluate(
        mut self,
        x: Tensor[DType.float64],
        y: Tensor[DType.float64]
    ) raises -> Float64:
        var predictions = self.predict(x)
        return self.get_accuracy(predictions, y)

    ########### helper stuff ################

    @always_inline
    @staticmethod
    fn _exp_tensor(Z: Tensor[DType.float64]) -> Tensor[DType.float64]:
        return Self._elemwise_math[exp](Z)

    @always_inline
    @staticmethod
    fn _elemwise_math[ 
        func: fn[dtype: DType, width: Int](SIMD[dtype, width]) -> SIMD[dtype, width]
    ](mat_in: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var mat_out = mat_in
        var size = mat_in.num_elements()
        if size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat_out.store(
                    idx,
                    func(mat_in.load(idx))
                )
            vectorize[math_vectorize, 1](size)
        else:
            @parameter
            fn math_vectorize_parallelize(i: Int):
                mat_out.store(
                    i,
                    func(mat_in.load(i))
                )
            parallelize[math_vectorize_parallelize](size, num_logical_cores())
        return mat_out^

    @always_inline
    @staticmethod
    fn _sum(z: Tensor[DType.float64]) -> Float64:
        try:
            return sum(
                Buffer[DType.float64](
                    z.unsafe_ptr(), 
                    z.num_elements()
                )
            )
        except:
            return 0.0

    @always_inline
    @staticmethod
    fn _max(z: Tensor[DType.float64]) -> Float64:
        try:
            return max(
                Buffer[DType.float64](
                    z.unsafe_ptr(), 
                    z.num_elements()
                )
            )
        except:
            return 0.0

    @always_inline
    @staticmethod
    fn _sum_tensor(z: Tensor[DType.float64], axis: Int = 0, keepdim: Bool = True) -> Tensor[DType.float64]:
        var shape = z.shape()
        var result = Tensor[DType.float64](TensorShape(0, 0)) # dummy tensor
        if axis == 0:
            if keepdim:
                result = Tensor[DType.float64](TensorShape(1, shape[1]))
            else:
                result = Tensor[DType.float64](TensorShape(shape[1]))     
            if shape[1] < 768:
                for j in range(shape[1]):
                    var sum: Float64 = 0.0
                    for i in range(shape[0]):
                        sum += z[i, j]
                    result.store(j, sum)
            else:
                @parameter
                fn p0(j: Int):
                    var sum: Float64 = 0.0
                    for i in range(shape[0]):
                        sum += z[i, j]
                    result.store(j, sum)
                parallelize[p0](shape[1], num_logical_cores())      
        elif axis == 1:
            if keepdim:
                result = Tensor[DType.float64](TensorShape(shape[0], 1))
            else:
                result = Tensor[DType.float64](TensorShape(shape[0]))
            if shape[0] < 768:
                for i in range(shape[0]):
                    var sum: Float64 = 0.0
                    for j in range(shape[1]):
                        sum += z[i, j]
                    result.store(i, sum)
            else:
                @parameter
                fn p1(i: Int):
                    var sum: Float64 = 0.0
                    for j in range(shape[1]):
                        sum += z[i, j]
                    result.store(i, sum)
                parallelize[p1](shape[0], num_logical_cores())
        return result^


    @staticmethod
    fn _transpose_tensor(x: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var shape = x.shape()
        var result = Tensor[DType.float64](TensorShape(shape[1], shape[0]))
        if x.num_elements() < 768:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    result[VariadicList(j, i)] = x[i, j]
        else:
            @parameter
            fn transpose_parallel(idx: Int):
                var i = idx // shape[1]
                var j = idx % shape[1]
                result[VariadicList(j, i)] = x[i, j]
            parallelize[transpose_parallel](shape[0] * shape[1], num_logical_cores())
        return result^

    @staticmethod
    fn _matmul(A: Tensor[DType.float64], B: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var m = A.shape()[0]
        var k = A.shape()[1]
        var n = B.shape()[1]
        var result = Tensor[DType.float64](TensorShape(m, n))
        if m * n < 768:
            for i in range(m):
                for j in range(n):
                    var sum: Float64 = 0.0
                    for p in range(k):
                        sum += A[i * k + p] * B[p * n + j]
                    result[i * n + j] = sum
        else:
            @parameter
            fn compute_element(idx: Int):
                var i = idx // n
                var j = idx % n
                var sum: Float64 = 0.0
                for p in range(k):
                    sum += A[i * k + p] * B[p * n + j]
                result[i * n + j] = sum  
            parallelize[compute_element](m * n, num_logical_cores())
        return result^

    @staticmethod
    fn add_broadcast(inout matrix: Tensor[DType.float64], bias: Tensor[DType.float64]):
        var shape = matrix.shape()
        if matrix.num_elements() < 768:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    matrix[i * shape[1] + j] += bias[i]
        else:
            @parameter
            fn add_bias(idx: Int):
                var i = idx // shape[1]
                var j = idx % shape[1]
                matrix[i * shape[1] + j] += bias[i]     
            parallelize[add_bias](matrix.num_elements(), num_logical_cores())