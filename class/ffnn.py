import numpy as np
from layer import Layer

class FeedForwardNeuralNetwork:
	def __init__(self):
		self.input_layer = []
		self.hidden_layer = []
		self.output_layer = []
		self.n_hidden_layer = 0
		self.activation_value = []
	
	def add_layer(self, layerType, input_size, n_neuron, activation, input_data, weights, bias, output, max_sse):
		if (layerType == "input_layer") :
			self.input_layer.append(Layer(input_size, n_neuron, activation, input_data, weights, bias, output, max_sse))
		elif (layerType == "hidden_layer") :
			self.hidden_layer.append(Layer(input_size, n_neuron, activation, input_data, weights, bias, output, max_sse))
			self.n_hidden_layer += 1
		elif (layerType == "output_layer") :
			self.output_layer.append(Layer(input_size, n_neuron, activation, input_data, weights, bias, output, max_sse))

	def forward_propagation(self, n_instance):
		for i in range (n_instance):
			print("Input layer")
			input_data = self.input_layer[0].input_data

			print("Input data yang ke-" + str(i+1) + " berupa" , input_data[0])
			print("=========================================================")

			for j in range (self.n_hidden_layer) :
				print("Hidden layer yang ke-" + str(j+1))

				print("Berikut informasi yang terdapat pada hidden layer tersebut")
				
				print("Weight = ")
				weight = self.hidden_layer[j].weights
				print(weight)

				print("Bias = ")
				bias = self.hidden_layer[j].bias
				print(bias)

				print("=========================================================")
				print("Diperoleh nilai sigma = ")
				if (j == 0) :
					sigma = np.dot(input_data, weight) + bias
				else :
					sigma = np.dot(activation_value, weight) + bias
				print(sigma)

				print("Diperoleh nilai fungsi aktivasi = ")
				activation_value = (self.hidden_layer[j].activation(sigma))
				print(list(activation_value[0]))
				print("=========================================================")

			self.activation_value.append(list(activation_value[0]))

	def printListActivationValue(self) :
		print("Berikut nilai output yang diperoleh dari input yang telah diberikan")
		print(self.activation_value)
		print("=========================================================")
	
	def accuracy(self) :
		expectOutput = self.output_layer[0].output
		nTotal = len(expectOutput)
		nAccurate = 0
		for i in range (nTotal) :
			for j in range (len(expectOutput[i])) :
				if (float(expectOutput[i][j]) == float(self.activation_value[i][j])) :
					nAccurate = nAccurate + 1
		
		percentAccurate = (nAccurate / nTotal ) * 100
		print("Akurasi yang diperoleh senilai " + str(percentAccurate) + "%")
