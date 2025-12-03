import numpy as np
from gensamples import getsamples     # or: from gensamples import getsamples
from a5ece457b import DecTree, kNN, NeuralNetwork


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def test_models_on_gen():
    print("\n=== Loading XOR Data from gen.py ===")
    X, y = getsamples()

    # Shuffle so classes are mixed
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    # Train/test split (70/30)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\n=== Decision Tree ===")
    tree = DecTree()
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    print("Test Accuracy:", accuracy(y_test, pred))

    print("\n=== kNN ===")
    knn = kNN()
    knn.k = 5
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print("Test Accuracy:", accuracy(y_test, pred))

    print("\n=== Neural Network ===")
    nn = NeuralNetwork()
    nn.hidden_size = 10
    nn.learning_rate = 0.1
    nn.epochs = 3000             # XOR requires more training!
    nn.fit(X_train, y_train)
    pred = nn.predict(X_test)
    print("Test Accuracy:", accuracy(y_test, pred))


if __name__ == "__main__":
    test_models_on_gen()
