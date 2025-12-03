"""
======================================
ECE457b Assignment 5
======================================
"""

# Credit ChatGPT

import numpy as np

class DecTree:
    class Node:
        def __init__(self, *, feature_index=None, threshold=None,
                     left=None, right=None, value=None):
            self.feature_index = feature_index  
            self.threshold = threshold          
            self.left = left                    
            self.right = right                  
            self.value = value                  

        def is_leaf(self):
            return self.value is not None

    def __init__(self):
        self.root = None
        self.max_depth = 5

    def _gini(self, y):
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None, 0.0

        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            x_column = X[:, feature_index]
            sorted_indices = np.argsort(x_column)
            x_sorted = x_column[sorted_indices]
            y_sorted = y[sorted_indices]

            unique_vals = np.unique(x_sorted)
            if unique_vals.size == 1:
                continue 

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = x_sorted <= threshold
                right_mask = ~left_mask

                y_left = y_sorted[left_mask]
                y_right = y_sorted[right_mask]

                if y_left.size == 0 or y_right.size == 0:
                    continue

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)

                w_left = y_left.size / n_samples
                w_right = y_right.size / n_samples

                child_gini = w_left * gini_left + w_right * gini_right
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]

        if depth >= self.max_depth or unique_classes.size == 1 or num_samples == 0:
            return self.Node(value=majority_class)

        feature_index, threshold, gain = self._best_split(X, y)

        if feature_index is None or gain <= 1e-12:
            return self.Node(value=majority_class)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return self.Node(feature_index=feature_index,
                         threshold=threshold,
                         left=left_child,
                         right=right_child)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)

    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


class kNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = 5 

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _predict_one(self, x):
        diffs = self.X_train - x
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        k = min(self.k, len(dists))

        nn_indices = np.argpartition(dists, kth=k - 1)[:k]
        nn_labels = self.y_train[nn_indices]

        values, counts = np.unique(nn_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


class NeuralNetwork:
    def __init__(self):
        self.hidden_size = 10
        self.learning_rate = 0.1
        self.epochs = 1000

        self.W1 = None  
        self.b1 = None  
        self.W2 = None  
        self.b2 = None  

    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    @staticmethod
    def _tanh_deriv(z):
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _initialize_params(self):
        rng = np.random.default_rng()
        self.W1 = rng.normal(loc=0.0, scale=0.1,
                             size=(2, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        self.W2 = rng.normal(loc=0.0, scale=0.1,
                             size=(self.hidden_size, 1))
        self.b2 = np.zeros(1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)  # column vector

        n_samples, n_features = X.shape
        assert n_features == 2, "Input must have 2 features."

        self._initialize_params()

        lr = self.learning_rate
        epochs = self.epochs

        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1  
            a1 = self._tanh(z1)         

            z2 = a1 @ self.W2 + self.b2  
            a2 = self._sigmoid(z2)       

            epsilon = 1e-12  
            a2_clipped = np.clip(a2, epsilon, 1 - epsilon)
            dL_da2 = -(y / a2_clipped - (1 - y) / (1 - a2_clipped))
            d_a2_dz2 = a2 * (1.0 - a2)
            dL_dz2 = dL_da2 * d_a2_dz2  

            dL_dW2 = a1.T @ dL_dz2 / n_samples            
            dL_db2 = np.mean(dL_dz2, axis=0)              

            dL_da1 = dL_dz2 @ self.W2.T                   
            d_a1_dz1 = self._tanh_deriv(z1)               
            dL_dz1 = dL_da1 * d_a1_dz1                     

            dL_dW1 = X.T @ dL_dz1 / n_samples              
            dL_db1 = np.mean(dL_dz1, axis=0)               

            self.W1 -= lr * dL_dW1
            self.b1 -= lr * dL_db1
            self.W2 -= lr * dL_dW2
            self.b2 -= lr * dL_db2

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        z1 = X @ self.W1 + self.b1
        a1 = self._tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)  

        y_pred = (a2 >= 0.5).astype(int).reshape(-1)
        return y_pred
