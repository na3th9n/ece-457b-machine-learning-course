import numpy as np
from a5ece457b import DecTree, kNN, NeuralNetwork


def generate_synthetic_data(n=300, seed=0):
    """
    Simple 2-class dataset.
    Class 0: Gaussian centered at (-1, -1)
    Class 1: Gaussian centered at (1, 1)
    """
    rng = np.random.default_rng(seed)

    X0 = rng.normal(loc=[-1, -1], scale=0.5, size=(n // 2, 2))
    X1 = rng.normal(loc=[1, 1], scale=0.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])

    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def test_decision_tree():
    print("\n=== Testing Decision Tree ===")
    X, y = generate_synthetic_data(300, seed=1)

    tree = DecTree()
    tree.fit(X, y)
    pred = tree.predict(X)
    acc = accuracy(y, pred)

    print(f"Decision Tree Accuracy: {acc:.3f}")


def test_knn():
    print("\n=== Testing kNN ===")
    X, y = generate_synthetic_data(300, seed=2)

    model = kNN()
    model.k = 5
    model.fit(X, y)
    pred = model.predict(X)
    acc = accuracy(y, pred)

    print(f"kNN Accuracy: {acc:.3f}")


def test_neural_network():
    print("\n=== Testing Neural Network ===")
    X, y = generate_synthetic_data(300, seed=3)

    nn = NeuralNetwork()
    nn.hidden_size = 10
    nn.learning_rate = 0.1
    nn.epochs = 1500  # adjust if learning slower/faster

    nn.fit(X, y)
    pred = nn.predict(X)
    acc = accuracy(y, pred)

    print(f"Neural Network Accuracy: {acc:.3f}")


if __name__ == "__main__":
    test_decision_tree()
    test_knn()
    test_neural_network()
