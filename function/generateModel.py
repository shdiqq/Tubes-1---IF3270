import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'class')
sys.path.append(mymodule_dir)

import json
import numpy as np
from ffnn import FeedForwardNeuralNetwork

def generate_model(filename):
	model = open(filename)
	model_data = json.loads(model.read())
	model.close()

	ffnn = FeedForwardNeuralNetwork()

	for x in model_data: # case dan expect
		for y in model_data[str(x)] : # model, input, weights, output, max_sse
			if (str(y) == "model") :
				for z in (model_data[str(x)])[str(y)] : #input_size, layers
					if (str(z) == "layers" ) : # array of list
						activation_function = []
						n_neuron = []
						for j in ((model_data[str(x)])[str(y)])[str(z)] :
							activation_function.append(j['activation_function'])
							n_neuron.append(j['number_of_neurons'])
					else : # "input_size"
						input_size = ((model_data[str(x)])[str(y)])[str(z)]
			elif (str(y) == 'input'):
					input_data = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'weights'):
				bias = []
				weights = []
				for z in range(len(((model_data[str(x)])[str(y)]))) :
					bias_temp = []
					weights_temp = []
					for i in range(len((((model_data[str(x)])[str(y)]))[z])) :
						if ( i == 0 ):
							bias_temp.append(((((model_data[str(x)])[str(y)][z][i]))))
						else :
							weights_temp.append(((((model_data[str(x)])[str(y)][z][i]))))
					bias.append(bias_temp)
					weights.append(weights_temp)
			elif (str(y) == 'output'):
				output = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'max_sse'):
				max_sse = ((model_data[str(x)])[str(y)])

	# Input Layer
	ffnn.add_layer( 'input_layer', input_size=int(input_size), n_neuron=None, activation='None', input_data=(input_data), weights=None, bias=None, output=None, max_sse=None )

	# Hidden Layer
	for i in range (len(weights)) :
		ffnn.add_layer( 'hidden_layer', input_size=None, n_neuron=int(n_neuron[i]), activation=str(activation_function[i]), input_data=None, weights=(weights[i]), bias=(bias[i]), output=None, max_sse=None )

	# Output Layer
	ffnn.add_layer( 'output_layer', input_size=None, n_neuron=None, activation='None', input_data=None, weights=None, bias=None, output=output, max_sse=max_sse )

	return ffnn
