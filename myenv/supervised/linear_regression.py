import numpy as np 
import matplotlib.pyplot as plt

def meanSquaredError(y_t, y_p):
    squaredDiff = (y_t - y_p) ** 2
    cost = np.sum(squaredDiff / (2 * len(y_t)))
    return cost


def gradientDescent(y,data,iterations):
    w = 0.5
    b = 0.8
    alpha = 0.001
    n = float(len(data))
    weights = []
    costs = []
   
    for i in range(iterations):
        yp = w * data + b
        yy = (y - yp)
        current_cost = meanSquaredError(y, yp)
        costs.append(current_cost)
        weights.append(w)

        wd = -(2/n) * np.sum((yy)*data)
        bd = -(2/n) * np.sum(yy)
        
        w = w - (alpha * wd)
        b = b - (alpha * bd)

        print(f"iteration {i+1}: cost {current_cost},weight {w} ,bias {b}")



    plt.figure(figsize = (8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("costs vs weights")
    plt.ylabel("costs")
    plt.xlabel("weight")
    plt.show()

    return w , b


def main():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.rand(100, 1)
    estimated_w , estimated_b  = gradientDescent(Y,X,10000)
    print(f"Estimated Weight: {estimated_w} , Estimated Bias: {estimated_b}")
    YP = estimated_w * X + estimated_b
    plt.figure(figsize = (8,6))
    plt.plot([min(X),max(X)],[min(YP),max(YP)], color = 'blue',markersize=10, linestyle='dashed')
    plt.scatter(X,Y,marker='o', color='red')
    plt.title("costs vs weights")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()
    
main()