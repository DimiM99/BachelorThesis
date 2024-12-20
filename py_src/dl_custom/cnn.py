import numpy as np
import pandas as pd

class SimpleNN:
    def __init__(self, input_size=784, hidden_size=10, output_size=10, debug=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.debug = debug
        self.W1, self.b1, self.W2, self.b2 = self._init_params()

    def _init_params(self):
        W1 = np.random.rand(self.hidden_size, self.input_size) - 0.5
        b1 = np.random.rand(self.hidden_size, 1) - 0.5
        W2 = np.random.rand(self.output_size, self.hidden_size) - 0.5
        b2 = np.random.rand(self.output_size, 1) - 0.5
        return W1, b1, W2, b2

    @staticmethod
    def relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def relu_deriv(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        return np.exp(Z) / sum(np.exp(Z))

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def forward(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y, m):
        one_hot_Y = self.one_hot(Y)
        dZ2 = self.A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.relu_deriv(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def fit(self, X_train, Y_train, learning_rate=0.1, epochs=100):
        m = X_train.shape[1]

        for epoch in range(epochs):
            # Forward propagation
            A2 = self.forward(X_train)

            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X_train, Y_train, m)

            # Update parameters
            self.update_params(dW1, db1, dW2, db2, learning_rate)

            if self.debug and epoch % 10 == 0:
                predictions = self.get_predictions(A2)
                accuracy = self.get_accuracy(predictions, Y_train)
                print(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        A2 = self.forward(X)
        return self.get_predictions(A2)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        return self.get_accuracy(predictions, Y)

# Example usage:
def prepare_data(data_path):
    print("Loading MNIST Digits Dataset...")
    print("Checking if dataset is already downloaded...")
    try:
        data = pd.read_csv("mnist.csv")
        print("Dataset loaded from a local copy.")
    except FileNotFoundError:
        print("Dataset is not found. Downloading from the GCS Bucket... (may take a while)")
        data =pd.read_csv("https://storage.googleapis.com/mnist-test-mojo-ba/mnist.csv")
        data.to_csv("mnist.csv", index=False)

    m, n = data.shape

    # Split into train and validation
    split_idx = 1000

    # Prepare validation data
    data_dev = data[0:split_idx].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    # Prepare training data
    data_train = data[split_idx:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    return X_train, Y_train, X_dev, Y_dev

if __name__ == "__main__":
    # Load and prepare data
    X_train, Y_train, X_val, Y_val = prepare_data('https://storage.googleapis.com/mnist-test-mojo-ba/mnist.csv')

    # Create and train model
    model = SimpleNN(input_size=784, hidden_size=10, output_size=10)
    model.fit(X_train, Y_train, learning_rate=0.1, epochs=500)

    # Evaluate on validation set
    val_accuracy = model.evaluate(X_val, Y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
