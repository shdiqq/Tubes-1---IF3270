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

	for layer in model_data:
		typeLayer = model_data[str(layer)]
		if (str(layer) == 'input_layer') :
			ffnn.add_layer(str(layer), n_neuron=int(typeLayer['n_neuron']), X=np.array(typeLayer['X']), weights=None, bias=np.array(typeLayer['bias']), activation=typeLayer['activation'])
		elif (str(layer) == 'hidden_layer') :
			ffnn.add_layer(str(layer), n_neuron=int(typeLayer['n_neuron']), X=None, weights=np.array(typeLayer['weights']), bias=np.array(typeLayer['bias']), activation=typeLayer['activation'])
		elif (str(layer) == 'output_layer') :
			ffnn.add_layer(str(layer), n_neuron=int(typeLayer['n_neuron']), X=None, weights=np.array(typeLayer['weights']), bias=None, activation=typeLayer['activation'])

	return ffnn
