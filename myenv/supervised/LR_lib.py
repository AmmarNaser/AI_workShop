import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.rand(100, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, Y)
    
    # Make predictions
    Y_pred = model.predict(X)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(Y, Y_pred)
    
    print(f"Estimated Weight: {model.coef_[0][0]}, Estimated Bias: {model.intercept_[0]}")
    print(f"Mean Squared Error: {mse}")
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y_pred, color='blue', linestyle='dashed')
    plt.scatter(X, Y, marker='o', color='red')
    plt.title("Regression Line")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()

main()
