import numpy as np

# Sigmoid function

def nonlin(x, deriv=False):
	if (deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

# Input dataset

input_set = np.array([[0,0,1],
					[0,1,1],
					[1,0,1],
					[1,1,1]])

target_set = np.array([[0],
						[1],
						[1],
						[0]])

np.random.seed(1)

# Randomize initial weight
layer_1_synaptic_weights = 2*np.random.random((3,4)) - 1
layer_2_synaptic_weights = 2*np.random.random((4,1)) - 1

for j in range(60000):
	
	# feed forward through layers 0,1, and 2
	input_layer = input_set
	layer_1 = nonlin(np.dot(input_layer, layer_1_synaptic_weights))
	layer_2 = nonlin(np.dot(layer_1, layer_2_synaptic_weights))

	# Cost function
	layer_2_error = target_set - layer_2

	if (j% 10000) == 0:
		print("Error: " + str(np.mean(np.abs(layer_2_error))))

	# in what direction is the target value
	# were we really sure? if so, don't change it too much
	layer_2_delta = layer_2_error * nonlin(layer_2, True)

	# how much did the layer 1 value contribute to the layer 2 error (according to the weights)?
	layer_1_error = layer_2_delta.dot(layer_2_synaptic_weights.T)

	# in what direction was the target layer 1
	layer_1_delta = layer_1_error * nonlin(layer_1, True)

	layer_2_synaptic_weights = layer_2_synaptic_weights + layer_1.T.dot(layer_2_delta)
	layer_1_synaptic_weights = layer_1_synaptic_weights + input_layer.T.dot(layer_1_delta)


