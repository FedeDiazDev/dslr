import numpy as np
import pandas as pd
import sys
import json

# -----------------------
# Sigmoid
# -----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# Coste (opcional)
# -----------------------
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    return -(1/m) * np.sum(
        y*np.log(h + epsilon) + (1-y)*np.log(1-h + epsilon)
    )

# -----------------------
# Gradient Descent
# -----------------------
def gradient_descent(X, y, theta, alpha, iters):
    m = len(y)
    costs = []

    for _ in range(iters):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        costs.append(cost)

    return theta, costs

# -----------------------
# Cargar datos correctamente
# -----------------------
def load_data(path):
    df = pd.read_csv(path)

    y = df["Hogwarts House"]

    df = df.drop(columns=[
        "Index",
        "Hogwarts House",
        "First Name",
        "Last Name",
        "Birthday",
    ])
    df["Best Hand"] = df["Best Hand"].map({'Left': 0, 'Right': 1})

    #X = df.apply(pd.to_numeric, errors='coerce')
    X = df.fillna(df.median())

    return X.values, y.values

# -----------------------
# Normalizar
# -----------------------
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std

# -----------------------
# One-vs-All
# -----------------------
def train_ova(X, y, classes, alpha=0.1, iters=1000, debug=True):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]

    all_theta = np.zeros((len(classes), n + 1))
    all_costs = []

    for i, c in enumerate(classes):
        print(f"Entrenando clase {c}")

        y_binary = (y == c).astype(int)
        theta = np.zeros(n + 1)

        theta, costs = gradient_descent(X, y_binary, theta, alpha, iters)

        if debug:
            print(f"Último coste clase {c}: {costs[-1]}")

        all_theta[i] = theta
        all_costs.append(costs)

    return all_theta, all_costs

# -----------------------
# Guardar modelo
# -----------------------
def save_model(all_theta, mean, std, classes):
    model = {
        "theta": all_theta.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "classes": classes.tolist()
    }

    with open("outputs/model.json", "w") as f:
        json.dump(model, f)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    X, y = load_data(sys.argv[1])

    # Orden fijo (IMPORTANTE)
    classes = np.array(["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"])

    # Normalizar
    X, mean, std = normalize(X)

    # Entrenar
    all_theta, _ = train_ova(X, y, classes)

    # Guardar modelo
    save_model(all_theta, mean, std, classes)

    print("Modelo guardado en model.json")
