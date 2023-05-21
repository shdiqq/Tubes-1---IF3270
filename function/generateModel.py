import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'class')
sys.path.append(mymodule_dir)

import json
import decimal
from ffnn import FeedForwardNeuralNetwork

def generate_model(filePath):

	try:
		model = open(filePath)
	except FileNotFoundError:
		return False
	
	model_data = json.loads(model.read())
	model.close()

	#set variable
	activation_function = []
	n_neuron = []
	input_size = None
	input_data = None
	bias = []
	weights = []
	output = None
	max_sse = None

	for x in model_data: # case dan expect
		for y in model_data[str(x)] : # model, input, weights, output, max_sse
			if (str(y) == "model") :
				for z in (model_data[str(x)])[str(y)] : #input_size, layers
					if (str(z) == "layers" ) : # array of list
						for j in ((model_data[str(x)])[str(y)])[str(z)] :
							activation_function.append(j['activation_function'])
							n_neuron.append(j['number_of_neurons'])
					else : # "input_size"
						input_size = ((model_data[str(x)])[str(y)])[str(z)]
			elif (str(y) == 'input'):
					input_data = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'weights'):
				for z in range(len(((model_data[str(x)])[str(y)]))) :
					bias.append(model_data[str(x)][str(y)][z][0])
					weights.append(model_data[str(x)][str(y)][z][1:])
			elif (str(y) == 'output'):
				output = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'max_sse'):
				max_sse = ((model_data[str(x)])[str(y)])
				count_decimal_places = abs(decimal.Decimal(str(max_sse)).as_tuple().exponent)

	#create MiniBatchGradientDescent
	ffnn = FeedForwardNeuralNetwork(output, max_sse, count_decimal_places)

	#Add input, bias, and weights to layer
	# Input Layer
	ffnn.add_layer('input_layer', int(input_size), None, 'None', (input_data), None, None)

	# Hidden & Output Layer
	for i in range(len(bias)) :
		typeLayer = ""
		if (i != (len(bias)-1)):
			typeLayer = "hidden_layer"
		else:
			typeLayer = "output_layer"
		ffnn.add_layer(typeLayer, None, int(n_neuron[i]), str(activation_function[i]), None, weights[i], bias[i])

	return ffnn
