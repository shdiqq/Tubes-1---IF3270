import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from activation import linear, sigmoid, ReLU, softmax

class Layer:
	def __init__(self, n_neuron, activation, X=None, Y=None, weights=None, bias=None):
		activations = {
			'linear': linear,
			'sigmoid': sigmoid,
			'ReLU': ReLU,
			'softmax': softmax
		}

		if(n_neuron < 1):
			raise ValueError("Neuron tidak boleh lebih kecil dari 1")
		elif(activation not in ('linear', 'sigmoid', 'ReLU', 'softmax')):
			raise ValueError("Fungsi aktivasi tidak boleh selain dari salah satu diantara berikut, 'linear', 'sigmoid', 'ReLU', 'softmax'")
		else:
			self.n_neuron = n_neuron
			self.X = X
			self.Y = Y
			self.weights = weights
			self.bias = bias
			self.activation = activations[activation]
			self.activation_value = None