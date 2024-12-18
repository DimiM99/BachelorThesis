from collections import InlinedFixedVector, Dict
from utils import Span
from sklmini_mo.utility.matrix import Matrix
from sklmini_mo.utility.utils import euclidean_distance, manhattan_distance, le, CVP, partition

struct KNN:
    var k: Int
    var distance: fn(Matrix, Matrix) raises -> Float64
    var X_train: Matrix
    var y_train: Matrix

    fn __init__(inout self, k: Int = 3, metric: String = 'euc'):
        self.k = k
        if metric.lower() == 'man':
            self.distance = manhattan_distance
        else:
            self.distance = euclidean_distance
        self.X_train = Matrix(0, 0)
        self.y_train = Matrix(0, 0)

    fn fit(inout self, X: Matrix, y: Matrix) raises:
        self.X_train = X
        self.y_train = y

    fn predict(self, X: Matrix) raises -> Matrix:
        var y_pred = Matrix(X.height, 1)
        for i in range(X.height):
            y_pred[i,0] = self._predict(X[i])
        return y_pred^

    @always_inline
    fn _predict(self, x: Matrix) raises -> Float64:
        var distances = Matrix(1, self.X_train.height)
        var dis_indices = InlinedFixedVector[Int](capacity = distances.size)
        
        # Compute distances between x and all examples in the training set
        for i in range(distances.size):
            dis_indices.append(i)
            distances.data[i] = self.distance(x, self.X_train[i])
            
        # Sort distances such that first k elements are smallest
        partition[le](
            Span[Float64, __lifetime_of(distances)](
                unsafe_ptr= distances.data, 
                len= distances.size
            ), 
            dis_indices, 
            self.k
        )
        
        # Find most common class among k nearest neighbors using arrays
        var labels = InlinedFixedVector[Float64](self.k)
        var counts = InlinedFixedVector[Int](self.k)
        var most_common = self.y_train.data[dis_indices[0]]
        var max_count = 1
        
        # Add first label
        labels.append(most_common)
        counts.append(1)
        
        # Count remaining labels
        for i in range(1, self.k):
            var label = self.y_train.data[dis_indices[i]]
            var found = False
            
            # Check if we've seen this label before
            for j in range(len(labels)):
                if labels[j] == label:
                    counts[j] += 1
                    if counts[j] > max_count:
                        max_count = counts[j]
                        most_common = label
                    found = True
                    break
                    
            # New label
            if not found:
                labels.append(label)
                counts.append(1)
                
        return most_common