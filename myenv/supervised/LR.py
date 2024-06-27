import numpy as np
import matplotlib.pyplot as plt

def meanSquaredError(y_t, y_p):
    squaredDiff = (y_t - y_p) ** 2
    cost = np.mean(squaredDiff) / 2
    return cost

def gradientDescent(y, data, iterations):
    w = 0.5
    b = 1
    alpha = 0.01
    n = float(len(data))
    weights = []
    costs = []
    
    for i in range(iterations):
        yp = w * data + b
        current_cost = meanSquaredError(y, yp)
        costs.append(current_cost)
        weights.append(w)

        wd = -(2/n) * np.sum((y - yp) * data)
        bd = -(2/n) * np.sum(y - yp)
        
        w = w - (alpha * wd)
        b = b - (alpha * bd)

        print(f"Iteration {i+1}: Cost {current_cost}, Weight {w}, Bias {b}")

    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Costs vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()

    return w, b

def main():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.rand(100, 1)

    estimated_w, estimated_b = gradientDescent(Y, X, 100)

    print(f"Estimated Weight: {estimated_w}, Estimated Bias: {estimated_b}")

    YP = estimated_w * X + estimated_b
    plt.figure(figsize=(8, 6))
    plt.plot(X, YP, color='blue', markersize=10, linestyle='dashed')
    plt.scatter(X, Y, marker='o', color='red')
    plt.title("Regression Line")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()

main()
