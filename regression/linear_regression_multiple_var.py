import numpy as np
import matplotlib.pyplot as plt

def normalize(X, num_of_feature):
	means = []
	std_dev = []

	X_norm = X

	for i in range(0, num_of_feature):
		m = X[:, i].mean()
		std = X[:, i].std()
		means.append(m)
		std_dev.append(std)
		X_norm[:, i] = (X_norm[:, i] - m) / std


	return X_norm, means, std_dev

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


data = np.loadtxt('ex1data2.txt', delimiter=',')
num_of_feature = data.shape[1] - 1
train_X = data[:, :2]
train_Y = data[:, 2]
m = train_X.shape[0]

X_norm, means, std_dev = normalize(train_X, num_of_feature)

X = np.ones((m, num_of_feature + 1))
X[:, 1:(num_of_feature+1)] = X_norm
y = np.ones((m, 1))
y[:, 0] = train_Y


theta = np.random.randn(num_of_feature + 1, 1)

iterations = 2000
alpha = 0.01

cost_history, trained_theta = gradient_descent(X, y, theta, alpha, iterations)

plt.plot(xrange(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


#Predict price of a 1650 sq-ft 3 br house
price = np.array([1.0,   ((1650.0 - means[0]) / std_dev[0]), ((3 - means[1]) / std_dev[1])]).dot(trained_theta)
print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price)


