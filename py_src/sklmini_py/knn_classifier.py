import numpy as np

class KNeighborsClassifier:
    def __init__(self, k=3, metric='euc'):
        """KNN
        
        Parameters:
        -----------
        k : int, default=3
            Number of neighbors
        metric : str, default='euc'
            Distance metric ('euc' for Euclidean, 'man' for Manhattan)
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data"""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        """Predict using the k-nearest neighbors"""
        X = np.asarray(X)
        predictions = []
        
        for sample in X:
            if self.metric == 'man':
                distances = np.sum(np.abs(self.X_train - sample), axis=1)
            else:  # Euclidean distance
                distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Get most common class
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
            
        return np.array(predictions)
