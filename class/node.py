import numpy as np

class Node:
	def __init__(self, bias, weight, activation_function_name, activation_function):
		self.bias = bias
		self.weight = weight
		self.net = 0
		self.activation_function = activation_function
		self.activation_function_name = activation_function_name
		self.activation_function_value = 0

	def calculate_net(self, inputData):
		self.net = np.dot(self.weight, inputData)
		self.net += self.bias

	def activate_neuron(self, sum = None):
		if (self.activation_function_name != 'softmax'):
			self.activation_function_value = self.activation_function(self.net)
		else:
			self.activation_function_value = self.activation_function(self.net, sum)
