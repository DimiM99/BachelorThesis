from sklmini_mo.utility.matrix import Matrix
from sklmini_mo.utility.utils import CVM, sign, mse
from collections import Dict
import math
import time

struct LinearRegression(CVM):
    var lr: Float32
    var n_iters: Int
    var penalty: String
    var reg_alpha: Float32
    var l1_ratio: Float32
    var tol: Float32
    var batch_size: Int
    var random_state: Int
    var weights: Matrix
    var bias: Float32
    var X_mean: Matrix
    var X_std: Matrix
    var y_mean: Float32
    var y_std: Float32

    fn __init__(inout self, learning_rate: Float32 = 0.001, n_iters: Int = 1000, penalty: String = 'l2', reg_alpha: Float32 = 0.0, l1_ratio: Float32 = -1.0,
                tol: Float32 = 0.0, batch_size: Int = 0, random_state: Int = -1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = Matrix(0, 0)
        self.bias = 0.0
        self.X_mean = Matrix(0, 0)
        self.X_std = Matrix(0, 0)
        self.y_mean = 0.0
        self.y_std = 0.0

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        # Feature scaling
        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-8
        var X_scaled = (X - self.X_mean) / self.X_std
        
        # Target scaling
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        var y_scaled = (y - self.y_mean) / self.y_std
        
        # Initialize parameters
        self.weights = Matrix.zeros(X.width, 1)
        self.bias = 0.0
        
        var X_T = Matrix(0, 0)
        if self.batch_size <= 0:
            X_T = X_scaled.T()

        var l1_lambda = self.reg_alpha
        var l2_lambda = self.reg_alpha
        if self.l1_ratio >= 0.0:
            l1_lambda *= self.l1_ratio
            l2_lambda *= 1.0 - self.l1_ratio
        else:
            if self.penalty == 'l2':
                l1_lambda = 0.0
            else:
                l2_lambda = 0.0

        var prev_cost = math.inf[DType.float32]()
        
        # Gradient descent
        for _ in range(self.n_iters):
            var y_predicted = X_scaled * self.weights + self.bias

            if self.tol > 0.0:
                var cost = mse(y_scaled, y_predicted)
                if abs(prev_cost - cost) <= self.tol:
                    break
                prev_cost = cost
                
            # Update gradients using scaled values
            if self.batch_size <= 0:
                var dw = (X_T * (y_predicted - y_scaled)) / X.height
                if l1_lambda > 0.0:
                    dw += l1_lambda * sign(self.weights)
                if l2_lambda > 0.0:
                    dw += l2_lambda * self.weights
                var db = (y_predicted - y_scaled).sum() / X.height
                
                # Gradient descent step
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    fn predict(self, X: Matrix) raises -> Matrix:
        var X_scaled = (X - self.X_mean) / self.X_std
        var y_scaled = X_scaled * self.weights + self.bias
        return y_scaled * self.y_std + self.y_mean

    fn __init__(inout self, params: Dict[String, String]) raises:
        if 'learning_rate' in params:
            self.lr = atof(params['learning_rate']).cast[DType.float32]()
        else:
            self.lr = 0.01
        if 'n_iters' in params:
            self.n_iters = atol(params['n_iters'])
        else:
            self.n_iters = 1000
        if 'penalty' in params:
            self.penalty = params['penalty']
        else:
            self.penalty = 'l2'
        if 'reg_alpha' in params:
            self.reg_alpha = atof(params['reg_alpha']).cast[DType.float32]()
        else:
            self.reg_alpha = 0.0
        if 'l1_ratio' in params:
            self.l1_ratio = atof(params['l1_ratio']).cast[DType.float32]()
        else:
            self.l1_ratio = -1.0
        if 'tol' in params:
            self.tol = atof(params['tol']).cast[DType.float32]()
        else:
            self.tol = 0.0
        if 'batch_size' in params:
            self.batch_size = atol(params['batch_size'])
        else:
            self.batch_size = 0
        if 'random_state' in params:
            self.random_state = atol(params['random_state'])
        else:
            self.random_state = -1
        self.weights = Matrix(0, 0)
        self.bias = 0.0
        self.X_mean = Matrix(0, 0)
        self.X_std = Matrix(0, 0)
        self.y_mean = 0.0
        self.y_std = 0.0