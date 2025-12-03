import numpy as np
from a5ece457b import DecTree, kNN, NeuralNetwork


def harder_data(n=600, seed=0):
    """
    A more difficult 2-class dataset:
    - Class 0: ring (donut)
    - Class 1: inner circle
    Great for checking generalization.
    """
    rng = np.random.default_rng(seed)

    # Radii
    r_inner = rng.normal(0.7, 0.15, n // 2)
    r_outer = rng.normal(1.6, 0.15, n // 2)

    # Angles
    theta_inner = rng.uniform(0, 2 * np.pi, n // 2)
    theta_outer = rng.uniform(0, 2 * np.pi, n // 2)

    X_inner = np.stack([r_inner * np.cos(theta_inner),
                        r_inner * np.sin(theta_inner)], axis=1)
    X_outer = np.stack([r_outer * np.cos(theta_outer),
                        r_outer * np.sin(theta_outer)], axis=1)

    X = np.vstack([X_inner, X_outer])
    y = np.array([0]*(n//2) + [1]*(n//2))

    return X, y


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    M = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def run_test(model, X_train, y_train, X_test, y_test, name):
    print(f"\n=== {name} ===")

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Train Accuracy: {accuracy(y_train, train_pred):.3f}")
    print(f"Test Accuracy:  {accuracy(y_test, test_pred):.3f}")
    print("Confusion Matrix (test):\n", confusion_matrix(y_test, test_pred))


if __name__ == "__main__":
    # Generate dataset
    X, y = harder_data()

    # Train/test split
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Models
    run_test(DecTree(), X_train, y_train, X_test, y_test, "Decision Tree")
    
    knn = kNN()
    knn.k = 7
    run_test(knn, X_train, y_train, X_test, y_test, "kNN (k=7)")

    nn = NeuralNetwork()
    nn.hidden_size = 8
    nn.learning_rate = 0.1
    nn.epochs = 1500
    run_test(nn, X_train, y_train, X_test, y_test, "Neural Network")
