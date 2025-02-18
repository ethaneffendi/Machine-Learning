import numpy as np
import random
import matplotlib.pyplot as plt

# loss functions
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

# random linear classifier
def random_linear_classifier(dataset, k):
    randomParameters = []
    for i in range(k):
        theta_not = random.randint(0, 100)
        theta = np.array([])
        for i in range(dataset[0][0].shape[0]):
            theta = np.append(theta, [random.randint(0,100)])
        randomParameters.append((theta, theta_not))
    j_star = 0
    for j in range(k):
        if(training_set_error(randomParameters[j][0], randomParameters[j][1], dataset) < training_set_error(randomParameters[j_star][0], randomParameters[j_star][1], dataset)):
            j_star = j
    return (randomParameters[j_star][0], randomParameters[j_star][1])

#execution
positivePoints = [(np.array([[10],[90]]),1),(np.array([[15],[85]]),1),(np.array([[20],[87]]),1),(np.array([[30],[82]]),1),(np.array([[40],[88]]),1),(np.array([[37],[89]]),1),(np.array([[39],[95]]),1),(np.array([[60],[92]]),1),(np.array([[55],[88]]),1),(np.array([[80],[90]]),1),(np.array([[70],[88]]),1),(np.array([[90],[89]]),1)]
negativePoints = [(np.array([[10],[-90]]),-1),(np.array([[15],[-85]]),-1),(np.array([[20],[-87]]),-1),(np.array([[30],[-82]]),-1),(np.array([[40],[-88]]),-1),(np.array([[37],[-89]]),-1),(np.array([[39],[-95]]),-1),(np.array([[60],[-92]]),-1),(np.array([[55],[-88]]),-1),(np.array([[80],[-90]]),-1),(np.array([[70],[-88]]),-1),(np.array([[90],[-89]]),-1)]

theta, theta_not = random_linear_classifier(positivePoints + negativePoints, 10000000000)
print(theta)
print(theta_not)
print(training_set_error(theta, theta_not, positivePoints + negativePoints))

#visualization
positiveXCoords = np.array([point[0][0, 0] for point in positivePoints])
positiveYCoords = np.array([point[0][1, 0] for point in positivePoints])
negativeXCoords = np.array([point[0][0, 0] for point in negativePoints])
negativeYCoords = np.array([point[0][1, 0] for point in negativePoints])

plt.gca().set_xlim([0,100])
plt.gca().set_ylim([-100,100])

plt.scatter(positiveXCoords, positiveYCoords, color = 'blue', marker = "+")
plt.scatter(negativeXCoords, negativeYCoords, color = 'red', marker = "_")

separatorX = [0, 50, 100]
separatorY = [(-theta_not-(theta[0]*separatorX[0]))/theta[1],(-theta_not-(theta[0]*separatorX[1]))/theta[1],(-theta_not-(theta[0]*separatorX[2]))/theta[1]]

plt.plot(separatorX, separatorY, color = "black", marker = "o")

plt.show()
