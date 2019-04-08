import numpy as np 

# Sigmoid Function

def nonlin(x, deriv=False):
	if (deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# input dataset
input_set = np.array([ [0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1] ])

# output dataset
target_set = np.array([[0,0,1,1]]).T

# seed random numbers to make the output deterministic
np.random.seed(1)

# initialize synaptic weights randomly with mean 0
synaptic_weights = 2*np.random.random((3,1)) - 1

for iter in range(10000):
	
	# forward propogation
	input_layer = input_set
	output = nonlin(np.dot(input_layer, synaptic_weights))

	# how much did we miss? Loss function
	output_error = target_set - output

	# multiply how much we missed by the slope of the sigmoid at the values in layer 1
	output_delta = output_error * nonlin(output, True)

	synaptic_weights = synaptic_weights + np.dot(input_layer.T, output_delta)

print("Output After Training: ")
print(output)