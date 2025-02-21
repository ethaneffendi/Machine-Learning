import numpy as np
import matplotlib.pyplot as plt
import math

# loss function
def negative_log_likelihood_loss(g, a):
    return -(a*np.log10(g) + (1-a)*np.log10(1-g))

# sigmoid function
def sigmoid(z):
    return 1/(1+(np.exp(-z)))

# hypothesis
def hypothesis(x, theta, theta_not):
    return sigmoid(np.dot(theta.T, x)+theta_not)

# training set error
def training_set_error(theta, theta_not, dataset):
    sum_of_losses = 0
    for i in range(len(dataset)):
        sum_of_losses += negative_log_likelihood_loss(hypothesis(dataset[i][0], theta, theta_not), dataset[i][1])
    return sum_of_losses/len(dataset)

# regularizer (square of the norm of theta)
def regularizer(theta):
    sum = 0
    for i in range(theta.shape[0]):
        sum += (theta[i][0])**2
    return sum

# objective function
def J(theta, theta_not, dataset, lam):
    return training_set_error(theta, theta_not, dataset) + (lam/2) * regularizer(theta)

# gradient of J
def gradJ(theta, theta_not, dataset, lam):
    sum = np.zeros((dataset[0][0].shape[0], 1))
    for i in range(len(dataset)):
        sum = sum + (hypothesis(dataset[i][0], theta, theta_not) - dataset[i][1]) * dataset[i][0]
    return (sum * (1/len(dataset))) + (lam * theta)

# dJ/dtheta_not
def partial_derivative_for_theta_not(theta, theta_not, dataset):
    sum = 0
    for i in range(len(dataset)):
        sum += hypothesis(dataset[i][0], theta, theta_not) - dataset[i][1]
    return sum * (1/len(dataset))

# gradient descent
def gradient_descent(dataset, theta_initial, theta_not_initial, step_size, tolerance, lam):
    theta = theta_initial
    theta_not = theta_not_initial
    theta_previous = theta_initial
    theta_not_previous = theta_not_initial
    while abs(J(theta, theta_not, dataset, lam)-J(theta_previous, theta_not_previous, dataset, lam)) > tolerance:
        theta_previous = theta
        theta_not_previous = theta_not
        theta = theta_previous - step_size * gradJ(theta_previous, theta_not_previous, dataset, lam)
        theta_not = theta_not_previous - step_size * partial_derivative_for_theta_not(theta_previous, theta_not_previous, dataset)
    return (theta, theta_not)

# two-dimensional perceptron with Matplotlib visualization
def gradient_descent_with_visualization(positivePoints, negativePoints, theta_initial, theta_not_initial, step_size, tolerance, lam, plotXMin, plotXMax, plotYMin, plotYMax):
    dataset = positivePoints + negativePoints
    theta = theta_initial
    theta_not = theta_not_initial
    theta_previous = theta_initial
    theta_not_previous = theta_not_initial
    while abs(J(theta, theta_not, dataset, lam)-J(theta_previous, theta_not_previous, dataset, lam)) > tolerance:
        theta_previous = theta
        theta_not_previous = theta_not
        theta = theta_previous - step_size * gradJ(theta_previous, theta_not_previous, dataset, lam)
        theta_not = theta_not_previous - step_size * partial_derivative_for_theta_not(theta_previous, theta_not_previous, dataset)
    plt.gca().set_xlim([plotXMin, plotXMax])
    plt.gca().set_ylim([plotYMin, plotYMax])
    plt.scatter(np.array([point[0][0, 0] for point in positivePoints]), np.array([point[0][1, 0] for point in positivePoints]), color = 'blue', marker = "+")
    plt.scatter(np.array([point[0][0, 0] for point in negativePoints]), np.array([point[0][1, 0] for point in negativePoints]), color = 'red', marker = "_")
    lineXCoords = [plotXMin, (plotXMin+plotXMax)/2, plotXMax]
    plt.plot(lineXCoords, [(-theta_not-(theta[0][0]*lineXCoords[0]))/theta[1][0],(-theta_not-(theta[0][0]*lineXCoords[1]))/theta[1][0],(-theta_not-(theta[0][0]*lineXCoords[2]))/theta[1][0]])
    plt.show()
# dataset to try
# positivePoints = [(np.array([[-2],[3]]),1),(np.array([[0],[1]]),1),(np.array([[2],[-1]]),1)]
# negativePoints = [(np.array([[-2],[1]]),0),(np.array([[0],[-1]]),0),(np.array([[2],[-3]]),0)]
# initialTheta = np.array([[0.06991289], [0.07085633]])
