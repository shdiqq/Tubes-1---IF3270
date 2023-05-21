import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from activation import linear, sigmoid, relu, softmax
from node import Node

class Layer:
	def __init__(self, n_neuron, activation_function_name, weights, bias):
		activation_function = {
			'linear': linear,
			'sigmoid': sigmoid,
			'relu': relu,
			'softmax': softmax,
			'None' : None
		}
		self.n_neuron = n_neuron
		self.activation_function_name = activation_function_name
		self.activation_function = activation_function[activation_function_name]
		self.activation_function_value = []
		self.weights = weights
		self.bias = bias
		self.net = []
		self.nodes = []
		self.generate_nodes()

	def generate_nodes(self):
		for i in range(self.n_neuron):
			thisWeight = []
			for j in range(len(self.weights)):
				thisWeight.append(self.weights[j][i])
			node = Node(self.bias[i], thisWeight, self.activation_function_name, self.activation_function)
			self.nodes.append(node)