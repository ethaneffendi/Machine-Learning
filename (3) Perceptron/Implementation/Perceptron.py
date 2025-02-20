import numpy as np
import matplotlib.pyplot as plt

# loss function
def zero_one_loss(g, a):
    if g == a:
        return 0
    else:
        return 1

def squared_loss(g, a):
    return (g-a)**2

# sign function
def sign_function(input):
    if input > 0:
        return 1
    else:
        return -1

# hypothesis
def hypothesis(x, theta, theta_not):
    input = np.dot(theta.T, x) + theta_not
    return sign_function(input)

# training set error
def training_set_error(theta, theta_not, dataset):
    sum_of_losses = 0
    for i in range(len(dataset)):
        sum_of_losses += zero_one_loss(hypothesis(dataset[i][0], theta, theta_not), dataset[i][1])
    return sum_of_losses/len(dataset)

# perceptron
def perceptron(dataset, T):
    theta_not = 0
    theta = np.zeros((dataset[0][0].shape[0], 1))
    for j in range(T):
        for i in range(len(dataset)):
            if(dataset[i][1] * (np.dot(theta.T, dataset[i][0])+theta_not) <= 0):
                theta = np.add(theta, dataset[i][1] * dataset[i][0])
                theta_not = theta_not + dataset[i][1]
    return (theta, theta_not)

def perceptron_with_visualization(positivePoints, negativePoints, T, plotXMin = 0, plotXMax = 100, plotYMin = -100, plotYMax = 100):
    dataset = positivePoints + negativePoints
    theta_not = 0
    theta = np.zeros((dataset[0][0].shape[0], 1))
    fig, ax = plt.subplots()
    plt.gca().set_xlim([plotXMin, plotXMax])
    plt.gca().set_ylim([plotYMin, plotYMax])
    ax.scatter(np.array([point[0][0, 0] for point in positivePoints]), np.array([point[0][1, 0] for point in positivePoints]), color = 'blue', marker = "+")
    ax.scatter(np.array([point[0][0, 0] for point in negativePoints]), np.array([point[0][1, 0] for point in negativePoints]), color = 'red', marker = "_")
    lineXCoords = [plotXMin, (plotXMin+plotXMax)/2, plotXMax]
    lineYCoords = [1,1,1]
    ln, = ax.plot(lineXCoords, lineYCoords)
    for j in range(T):
        for i in range(len(dataset)):
            if(dataset[i][1] * (np.dot(theta.T, dataset[i][0])+theta_not) <= 0):
                theta = np.add(theta, dataset[i][1] * dataset[i][0])
                theta_not = theta_not + dataset[i][1]
                ax.cla()
                ax.scatter(np.array([point[0][0, 0] for point in positivePoints]), np.array([point[0][1, 0] for point in positivePoints]), color = 'blue', marker = "+")
                ax.scatter(np.array([point[0][0, 0] for point in negativePoints]), np.array([point[0][1, 0] for point in negativePoints]), color = 'red', marker = "_")
                ax.plot(lineXCoords, [(-theta_not-(theta[0][0]*lineXCoords[0]))/theta[1][0],(-theta_not-(theta[0][0]*lineXCoords[1]))/theta[1][0],(-theta_not-(theta[0][0]*lineXCoords[2]))/theta[1][0]])
                plt.gca().set_xlim([plotXMin, plotXMax])
                plt.gca().set_ylim([plotYMin, plotYMax])
                plt.draw()
                plt.pause(0.5)
    plt.show()





# dataset to try
# positivePoints = [(np.array([[-2],[3]]),1), (np.array([[0],[1]]),1), (np.array([[2],[-1]]),1)]
# negativePoints = [(np.array([[-2],[1]]),-1), (np.array([[0],[-1]]),-1), (np.array([[2],[-3]]),-1)]

# positivePoints = [(np.array([[0.5],[3]]),1), (np.array([[0.4],[2.7]]),1), (np.array([[1],[3.5]]),1)]
# negativePoints = [(np.array([[0.5],[2]]),-1), (np.array([[1],[2.5]]),-1), (np.array([[1.5],[3]]),-1)]
