import numpy as np
from layer import Layer

class FeedForwardNeuralNetwork:
	def __init__(self):
		self.input_layer = []
		self.hidden_layer = []
		self.output_layer = []
		self.n_input_layer = 0
		self.n_hidden_layer = 0
		self.n_output_layer = 0
		self.prediction = None
	
	def add_layer(self, layerType, activation, n_neuron, X=None, weights=None, bias=None):
		if (layerType == "input_layer") :
			self.input_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=X, weights=None, bias=bias))
			self.n_input_layer += 1
		elif (layerType == "hidden_layer") :
			self.hidden_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=None, weights=weights, bias=bias))
			self.n_hidden_layer += 1
		elif (layerType == "output_layer") :
			self.output_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=None, weights=weights, bias=None))
			self.n_output_layer += 1
