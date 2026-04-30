import numpy as np
import pandas as pd
import sys
import json

from pathlib import Path

from bonus.methods import batch_gradient_descent
from bonus.methods import stochastic_gradient_descent
from bonus.methods import mini_batch_gradient_descent

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
def train_ova(X, y, classes, method, debug=False, alpha=0.1, iters=1000, epochs=50, batch_size=32):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]

    all_theta = np.zeros((len(classes), n + 1))
    all_costs = []

    for i, c in enumerate(classes):
        print(f"Entrenando clase {c}")

        y_binary = (y == c).astype(int)
        theta = np.zeros(n + 1)

        match method:
            case "batch":
                theta, costs = batch_gradient_descent(X, y_binary, theta, alpha, iters, debug)
            case "sgd":
                alpha = 0.01
                theta = stochastic_gradient_descent(X, y_binary, theta, alpha, epochs, debug)
            case "mini-batch":
                alpha = 0.01
                theta = mini_batch_gradient_descent(X, y_binary, theta, alpha, epochs, batch_size, debug)
            case _:
                raise ValueError("Método no soportado")


        all_theta[i] = theta
        if method == "batch":
            all_costs.append(costs)

    return all_theta, all_costs

# -----------------------
# Guardar modelo
# -----------------------
def save_model(all_theta, mean, std, classes, method):
    model = {
        "theta": all_theta.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "classes": classes.tolist()
    }
    
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "outputs" / (method + "_model.json")

    with open(path, "w") as f:
        json.dump(model, f)

    print("Modelo guardado en " + method + "_model.json")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    len_args = len(sys.argv)

    debug = False
    match len_args:
        case 2:
            method = "batch"
        case 3:
            if (sys.argv[2] == "debug"):
                debug = True
            else:
                method = sys.argv[2]
        case 4:
            if (sys.argv[3] == "debug"):
                debug = True
            else:
                print("Uso: python logreg_train.py dataset_train.csv (method) (debug)")
                sys.exit(1)
            method = sys.argv[2]
        case _:
            print("Uso: python logreg_train.py dataset_train.csv (method)")
            sys.exit(1)

    file = sys.argv[1]

    X, y = load_data(file)

    # Orden fijo (IMPORTANTE)
    classes = np.array(["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"])

    # Normalizar
    X, mean, std = normalize(X)

    # Entrenar
    all_theta, _ = train_ova(X, y, classes, method, debug)

    # Guardar modelo
    save_model(all_theta, mean, std, classes, method)
