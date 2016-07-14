import numpy as np

def sigmoid(x):
	return 1 / (1+np.exp(-x))

def sigmoid_deriv(x):
	return x * (1 - x)

# dimen of X = 4,3
X = np.array([
		[0,0,1],
		[1,1,1],
		[1,0,1],
		[0,1,1],
	])

# dimen of Y = 4,1
y = np.array([
		[0],
		[1],
		[1],
		[0],
	])


np.random.seed(1)

# randomly initialize the synapses with mean 0
# dimension of W1 = 3,4 ; dimnesion of W2 = 4,1
W1 = 2 * np.random.random((3, 4)) - 1
W2 = 2 * np.random.random((4, 1)) - 1

for i in xrange(50000):
	# forward propagation
	A1 = X
	Z2 = np.dot(A1, W1)
	A2 = sigmoid(Z2)
	Z3 = np.dot(A2, W2)
	A3 = sigmoid(Z3) 

	# backpropagation
	error = A3 - y
	if (i % 5000 == 0):
		print "Error: ", np.mean(np.abs(error))
		
	delta2 = error * sigmoid_deriv(A3)
	dJdW2 = A2.T.dot(delta2)

	delta1 = delta2.dot(W2.T) * sigmoid_deriv(A2)
	dJdW1 = A1.T.dot(delta1)

	W1 -= dJdW1
	W2 -= dJdW2



test_data = np.array([
				[1, 0 ,0]
			])

result = sigmoid(np.dot(sigmoid(np.dot(test_data, W1)), W2))
print "Test Data: ", test_data
print "Result: ", result


