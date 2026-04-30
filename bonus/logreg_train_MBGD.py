import numpy as np
import pandas as pd
import sys
import json

# -----------------------
# Sigmoid
# -----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# Mini Batch Gradient Descent
# -----------------------------
def mini_batch_gradient_descent(X, y, theta, alpha, epochs, batch_size):
    m = len(y)

    for epoch in range(epochs):
        # data shuffle
        index = np.random.permutation(m) 
        X = X[index]
        y = y[index]

        # alpha = init_alpha / (1 + epoch * 0.1)

        for i in range(m):
            x_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            h = sigmoid(x_batch @ theta)

            gradient = ((1/len(y_batch)) * (x_batch.T @ (h - y_batch)))

            theta -= alpha * gradient

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")

    return theta

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
def train_ova(X, y, classes, alpha=0.01, epochs=50, batch_size=32):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]

    all_theta = np.zeros((len(classes), n + 1))

    for i, c in enumerate(classes):
        print(f"Entrenando clase {c}")

        y_binary = (y == c).astype(int)
        theta = np.zeros(n + 1)

        theta = mini_batch_gradient_descent(X, y_binary, theta, alpha, epochs, batch_size)

        all_theta[i] = theta

    return all_theta

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

    with open("outputs/MBGD_model.json", "w") as f:
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
