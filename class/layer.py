import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from activation import linear, sigmoid, relu, softmax

class Layer:
	def __init__(self, input_size, n_neuron, activation, input_data, weights, bias, output, max_sse):
		activations = {
			'linear': linear,
			'sigmoid': sigmoid,
			'relu': relu,
			'softmax': softmax,
			'None' : None
		}

		self.input_size = input_size
		self.n_neuron = n_neuron
		self.activation = activations[activation]
		self.input_data = input_data
		self.weights = weights
		self.bias = bias
		self.output = output
		self.max_sse = max_sse
		self.activation_value = None