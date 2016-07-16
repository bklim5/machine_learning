import numpy as np
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
	m = X.shape[0]
	hx = np.dot(X, theta)
	error = hx - y
	cost = (1.0/ (2*m)) * (error ** 2).sum()

	return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
	m = X.shape[0]
	J_history = np.zeros((num_iterations, 1))

	print "Initial theta: ", theta

	for i in xrange(num_iterations):
		predictions = np.dot(X, theta)

		theta = theta - (alpha / m) * np.dot((predictions - y).T, X).T

		J_history[i][0] = compute_cost(X, y, theta)

		if i % 100 == 0:
			print "Cost: " , J_history[i][0]
			print "Theta: " , theta


	return J_history, theta




# Training Data
# train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])

data = np.loadtxt('ex1data1.txt', delimiter=',')
num_of_feature = data.shape[1] - 1
train_X = data[:, 0]
train_Y = data[:, 1]
m = train_X.size
X = np.ones((m, num_of_feature + 1))
X[:, 1] = train_X
y = np.ones((m, 1))
y[:, 0] = train_Y

theta = np.random.randn(num_of_feature + 1, 1)

iterations = 2000
alpha = 0.01

cost_history, trained_theta = gradient_descent(X, y, theta, alpha, iterations)

plt.plot(xrange(iterations), cost_history)
plt.show()


plt.plot(train_X, train_Y, 'ro', label="Training data")
plt.plot(train_X, np.dot(X, trained_theta))

plt.show()




