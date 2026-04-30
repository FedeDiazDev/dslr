import numpy as np

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
# Batch Gradient Descent
# -----------------------
def batch_gradient_descent(X, y, theta, alpha, iters, debug):
    m = len(y)
    costs = []

    for _ in range(iters):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        costs.append(cost)

    if debug:
        print(f"Último coste: {costs[-1]}")

    return theta, costs


# -----------------------
# Stochastic Gradient Descent
# -----------------------
def stochastic_gradient_descent(X, y, theta, alpha, epochs, debug):
    m = len(y)

    for epoch in range(epochs):
        # data shuffle
        index = np.random.permutation(m) 

        X = X[index]
        y = y[index]

        # alpha = init_alpha / (1 + epoch * 0.1)

        for i in range(m):
            xi = X[i]
            yi = y[i]

            h = sigmoid(np.dot(xi, theta))
            error = h - yi
            theta -= alpha * error * xi # θ=θ−α⋅(h(xi)−yi)xi

        if debug and (epoch % 10 == 0):
            print(f"Epoch {epoch}")

    return theta


# -----------------------------
# Mini Batch Gradient Descent
# -----------------------------
def mini_batch_gradient_descent(X, y, theta, alpha, epochs, batch_size, debug):
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

        if debug and (epoch % 10 == 0):
            print(f"Epoch {epoch}")

    return theta

