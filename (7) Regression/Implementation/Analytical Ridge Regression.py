import numpy as np
import matplotlib.pyplot as plt


# analytical ridge regression
def analytical_ridge_regression(dataset, lam):
    n = len(dataset[0])
    W = np.zeros((len(dataset),n+1))
    T = np.zeros((len(dataset), 1))
    I = np.identity(W.shape[0])
    for i in range(len(dataset)):
        T[i][0] = dataset[i][1]
        for j in range(n):
            W[i][j] = dataset[i][0][j]
        W[i][n] = 1
    return np.linalg.inv(((W.T * W) + (n * lam * I))) * (W.T * T)

positivePoints = [(np.array([[-2],[3]]),1), (np.array([[0],[1]]),1), (np.array([[2],[-1]]),1)]
print(analytical_ridge_regression(positivePoints, 1))
