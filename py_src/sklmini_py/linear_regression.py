import numpy as np

class LinearRegression:
    """
    Linear Regression model with optional L1, L2, and Elastic Net regularization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, penalty=None, 
                 reg_alpha=0.0, l1_ratio=0.5, tol=0.0):
        """
        Parameters:
            learning_rate (float): Learning rate for gradient descent.
            n_iterations (int): Number of iterations for training.
            penalty (str): Type of regularization ('l1', 'l2', 'elasticnet', or None).
            reg_alpha (float): Regularization strength.
            l1_ratio (float): Mixing ratio between L1 and L2 for elastic net (0 < l1_ratio < 1).
            tol (float): Tolerance for early stopping based on cost improvement.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.penalty = penalty  # 'l1', 'l2', 'elasticnet', or None
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.tol = tol  # Tolerance for convergence
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.X_mean = None
        self.X_std = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        """
        # Feature scaling
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_scaled = (X - self.X_mean) / self.X_std
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for i in range(self.n_iterations):
            # Compute predictions
            y_pred = np.dot(X_scaled, self.weights) + self.bias

            # Compute errors
            errors = y_pred - y

            # Compute cost with regularization
            cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
            if self.penalty is not None:
                cost += self._compute_regularization_cost()
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_scaled.T, errors)
            db = (1 / n_samples) * np.sum(errors)
            if self.penalty is not None:
                dw += self._compute_regularization_gradient()

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if self.tol > 0 and i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Converged at iteration {i}")
                break

    def predict(self, X):
        """
        Predict target values for given input data.
        """
        # Apply the same scaling as during training
        X_scaled = (X - self.X_mean) / self.X_std
        y_pred = np.dot(X_scaled, self.weights) + self.bias
        return y_pred.flatten()

    def _compute_regularization_cost(self):
        """
        Compute the regularization term added to the cost function.
        """
        if self.penalty == 'l1':
            # L1 regularization (Lasso)
            reg_cost = self.reg_alpha * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            # L2 regularization (Ridge)
            reg_cost = (self.reg_alpha / 2) * np.sum(self.weights ** 2)
        elif self.penalty == 'elasticnet':
            # Elastic Net regularization
            l1_term = self.l1_ratio * np.sum(np.abs(self.weights))
            l2_term = (1 - self.l1_ratio) * np.sum(self.weights ** 2) / 2
            reg_cost = self.reg_alpha * (l1_term + l2_term)
        else:
            reg_cost = 0.0
        return reg_cost

    def _compute_regularization_gradient(self):
        """
        Compute the gradient of the regularization term.
        """
        if self.penalty == 'l1':
            # Gradient of L1 regularization
            reg_grad = self.reg_alpha * np.sign(self.weights)
        elif self.penalty == 'l2':
            # Gradient of L2 regularization
            reg_grad = self.reg_alpha * self.weights
        elif self.penalty == 'elasticnet':
            # Gradient of Elastic Net regularization
            l1_grad = self.l1_ratio * np.sign(self.weights)
            l2_grad = (1 - self.l1_ratio) * self.weights
            reg_grad = self.reg_alpha * (l1_grad + l2_grad)
        else:
            reg_grad = 0.0
        return reg_grad
