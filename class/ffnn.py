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
	
	def add_layer(self, layerType, n_neuron, activation, X=None, weights=None, bias=None):
		if (layerType == "input_layer") :
			self.input_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=X, weights=None, bias=bias))
			self.n_input_layer += 1
		elif (layerType == "hidden_layer") :
			self.hidden_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=None, weights=weights, bias=bias))
			self.n_hidden_layer += 1
		elif (layerType == "output_layer") :
			self.output_layer.append(Layer(n_neuron=n_neuron, activation=activation, X=None, weights=weights, bias=None))
			self.n_output_layer += 1

	def predict(self, n_instance):
		for i in range (n_instance):
			print("Input layer")
			listData = self.input_layer[0].X

			X = np.array([listData[i]])
			print("Input data yang ke - " + str(i+1) + " berupa" , X)

			for j in range (self.n_hidden_layer) :
				listWeight = self.hidden_layer[j].weights
				listBias = self.input_layer[j].bias
				print("Hidden layer yang ke - " + str(j+1))
				print("Berikut informasi yang terdapat pada hidden layer tersebut")
				print("Weight = ")
				print(listWeight)
				print("Bias = ")
				print(listBias)
				
				print("Diperoleh nilai sigma = ")
				sigma = np.dot(X, listWeight) + listBias
				print(sigma)
				print("Diperoleh nilai fungsi aktivasi = ")
				activation_value = self.hidden_layer[j].activation(sigma)
				print(activation_value)
			
			listWeight = self.output_layer[j].weights
			listBias = self.hidden_layer[j].bias
			print("Output layer")
			print("Berikut informasi yang terdapat pada output layer")
			print("Weight = ")
			print(listWeight)
			print("Bias = ")
			print(listBias)

			print("Diperoleh nilai sigma = ")
			sigma = np.dot(activation_value, listWeight) + listBias
			print(sigma)
			print("Diperoleh nilai fungsi aktivasi = ")
			activation_value = self.hidden_layer[j].activation(sigma)
			print(activation_value)
