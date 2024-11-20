import numpy as np
from numpy.random import seed

class KMeans:
    def __init__(self, K=5, init='kmeans++', max_iters=100, tol=1e-4, random_state=42):
        """ KMeans
    
        Parameters:
        -----------
        K : int, default=5
            Number of clusters
        init : str, default='kmeans++'
            Initialization method
        max_iters : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        random_state : int, default=42
            Random state for reproducibility
        """
        self.K = K
        self.init = init.lower()
        self.max_iters = max_iters
        self.tol = tol
        seed(random_state)
        self.clusters = []  # List of sample indices for each cluster
        self.centroids = None  # The centers for each cluster

    def _kmeans_plus_plus(self, data):
        """Initialize centroids using kmeans++ method"""
        # Randomly select first centroid
        self.centroids = np.zeros((self.K, data.shape[1]))
        self.centroids[0] = data[np.random.randint(data.shape[0])]

        # Select remaining centroids
        for i in range(1, self.K):
            # Compute distances to nearest centroid
            distances = np.array([min([np.sum((x - c) ** 2) for c in self.centroids[:i]]) 
                                for x in data])
            
            # Select next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            
            # Select the next centroid
            r = np.random.random()
            for j in range(len(cumulative_probs)):
                if r < cumulative_probs[j]:
                    self.centroids[i] = data[j]
                    break

    def predict(self, X):
        """Predict cluster labels for X"""
        if self.init == 'kmeans++':
            self._kmeans_plus_plus(X)
        else:
            # Random initialization
            idx = np.random.choice(X.shape[0], self.K, replace=False)
            self.centroids = X[idx]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = [[] for _ in range(self.K)]
            for idx, sample in enumerate(X):
                distances = [np.sum((sample - centroid) ** 2) for centroid in self.centroids]
                closest_centroid = np.argmin(distances)
                self.clusters[closest_centroid].append(idx)

            # Calculate new centroids
            old_centroids = self.centroids.copy()
            for i in range(self.K):
                if len(self.clusters[i]) > 0:
                    self.centroids[i] = np.mean(X[self.clusters[i]], axis=0)

            # Check convergence
            if np.sum((old_centroids - self.centroids) ** 2) < self.tol:
                break

        # Return cluster assignments
        labels = np.zeros(X.shape[0])
        for i, cluster in enumerate(self.clusters):
            labels[cluster] = i
        return labels
