import numpy as np
import pandas as pd
import sys
import json

from pathlib import Path

# -----------------------
# Sigmoid
# -----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# Cargar modelo
# -----------------------
def load_model(path):
    with open(path, "r") as f:
        model = json.load(f)

    return (
        np.array(model["theta"]),
        np.array(model["mean"]),
        np.array(model["std"]),
        np.array(model["classes"])
    )

# -----------------------
# Cargar datos correctamente
# -----------------------
def load_data(path):
    df = pd.read_csv(path)

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

    return X.values

# -----------------------
# Normalizar
# -----------------------
def normalize(X, mean, std):
    std[std == 0] = 1
    return (X - mean) / std

# -----------------------
# Predicción OvA
# -----------------------
def predict_ova(X, all_theta):
    m = X.shape[0]
    X = np.c_[np.ones((m, 1)), X]

    probs = sigmoid(X @ all_theta.T)
    return np.argmax(probs, axis=1)

# -----------------------
# Guardar CSV
# -----------------------
def save_predictions(predictions, classes):
    path = Path(__file__).resolve().parent / "outputs" / "houses.csv"
    with open(path, "w") as f:
        f.write("Index,Hogwarts House\n")

        for i, pred in enumerate(predictions):
            f.write(f"{i},{classes[pred]}\n")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python logreg_predict.py dataset_test.csv model.json")
        sys.exit(1)

    X = load_data(sys.argv[1])

    all_theta, mean, std, classes = load_model(sys.argv[2])

    # Normalizar igual que training
    X = normalize(X, mean, std)

    predictions = predict_ova(X, all_theta)

    save_predictions(predictions, classes)

    print("houses.csv generado ✔️")
