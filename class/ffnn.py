import networkx as nx
import matplotlib.pyplot as plt
from layer import Layer

class FeedForwardNeuralNetwork:
	def __init__(self, expect_output, max_sse, count_decimal_places):
		self.input_size						= 0
		self.input_layer					= []

		self.n_hidden_layer				= 0
		self.hidden_layer					= []

		self.output_layer					= []

		self.output								= []
		self.expect_output				= expect_output
		self.max_sse							= max_sse
		self.count_decimal_places = count_decimal_places
		print(count_decimal_places)

	def set_input_data(self, X_train):
		self.input_layer = X_train
		self.input_size = len(X_train[0])

	def add_layer(self, layerType, input_size, n_neuron, activation, input_data, weights, bias):
		if (layerType == "input_layer") :
			self.input_layer = input_data
			self.input_size = input_size
		elif (layerType == "hidden_layer") :
			self.hidden_layer.append(Layer(n_neuron, activation, weights, bias))
			self.n_hidden_layer += 1
		elif (layerType == "output_layer") :
			self.output_layer = Layer(n_neuron, activation, weights, bias)

	def reset_net_and_activation_function_value(self, typeLayer):
		if (typeLayer == 'hidden layer'):
			for i in range (self.n_hidden_layer):
				self.hidden_layer[i].net = []
				self.hidden_layer[i].activation_function_value = []
		else:
			self.output_layer.net = []
			self.output_layer.activation_function_value = []

	def forward_propagation(self):
		print("Proses Forward Propagation")
		for i in range (len(self.input_layer)):
			# hidden layers if any
			for j in range (self.n_hidden_layer) :
				print(f"Pada input hidden layer yang ke-{i+1}")
				# check is first hiddent layer or nots
				if (j == 0):
					prevLayer_activationValues = self.input_layer[i]
				else:
					prevLayer_activationValues = self.hidden_layer[j-1].activation_function_value

				# reset net and activation function value if any
				if (len(self.hidden_layer[j].net) != 0 ):
					self.reset_net_and_activation_function_value('hidden layer')

				# calculate net
				for node in self.hidden_layer[j].nodes:
					node.calculate_net(prevLayer_activationValues)
					self.hidden_layer[j].net.append(node.net)
				print(f"Nilai net pada layer ini adalah")
				print(f"{self.hidden_layer[j].net}")

				# calculate activation function value
				for node in self.hidden_layer[j].nodes:
					# check if the activation function is softmax
					if (node.activation_function_name == 'softmax'):
						node.activate_neuron(self.hidden_layer[j].net)
					else:
						node.activate_neuron()
					self.hidden_layer[j].activation_function_value.append(node.activation_function_value)
				print(f"Nilai fungsi aktivasi pada layer ini adalah")
				print(f"{self.hidden_layer[j].activation_function_value}")

			# output Layer
			print(f"Pada output layer")
			# check if there is only a hidden layer or not
			if (self.n_hidden_layer == 0):
				prevLayer_activationValues = self.input_layer[i]
			else:
				prevLayer_activationValues = self.hidden_layer[-1].activation_function_value

			# reset net and activation function value if any
			if (len(self.output_layer.net) != 0 ):
					self.reset_net_and_activation_function_value('output layer')

			# calculate net
			for node in self.output_layer.nodes:
				node.calculate_net(prevLayer_activationValues)
				self.output_layer.net.append(node.net)
			print(f"Nilai net pada layer ini adalah")
			print(f"{self.output_layer.net}")

			# calculate activation function value
			for node in self.output_layer.nodes:
					# check if the activation function is softmax
				if (node.activation_function_name == 'softmax'):
					node.activate_neuron(self.output_layer.net)
				else:
					node.activate_neuron()
				self.output_layer.activation_function_value.append(round(node.activation_function_value, self.count_decimal_places))
			print(f"Nilai fungsi aktivasi pada layer ini adalah")
			print(f"{self.output_layer.activation_function_value}")
			self.output.append(self.output_layer.activation_function_value)

	def accuracy(self):
		count = 0
		same = 0
		for i in range (len(self.expect_output)):
			for j in range (len(self.expect_output[i])):
				if (self.expect_output[i][j] == self.output[i][j]):
					same = same + 1
				elif (self.expect_output[i][j] + self.max_sse == self.output[i][j]):
					same = same + 1
				elif (self.expect_output[i][j] == self.output[i][j]) + self.max_sse:
					same = same + 1
				count = count + 1
		print(f"akurasi {(same/count) * 100}%")

	def information(self):
		print(f"Berikut informasi yang ada pada input layer")
		print(self.input_layer)

		if (self.n_hidden_layer != 0):
			print("")
			print(f"Berikut informasi yang ada pada hidden layer")
			for i in range(self.n_hidden_layer):
				print("")
				print(f"Pada hidden layer yang ke-{i+1} terdapat")
				for j in range (len(self.hidden_layer[i].nodes)):
					print(f"Pada node yang ke-{j+1} terdapat")
					print(f"Bias: {self.hidden_layer[i].nodes[j].bias}")
					print(f"Weight: {self.hidden_layer[i].nodes[j].weight}")

		print("")
		print(f"Berikut informasi yang ada pada output layer")
		for j in range (len(self.output_layer.nodes)):
			print(f"Pada node yang ke-{j+1} terdapat")
			print(f"Bias: {self.output_layer.nodes[j].bias}")
			print(f"Weight: {self.output_layer.nodes[j].weight}")

	def visualize(self):
		#set variable
		G = nx.Graph()
		pos = {}
		labels = {}
		counter = 1
		posCounter = 1
		inputNode = []
		hiddenNode = []
		outputNode = []
		biasNode = []

		#### 1.NODES ADJUSTMENT
		# 1. Nodes for input layer
		# + the input layer bias
		G.add_node(counter, label="bias")    
		pos[counter] = (0, posCounter)
		labels[counter] = "1"
		biasNode.append(counter)
		posCounter += 1
		counter += 1

		# + input nodes
		for i in range(self.input_size):
			G.add_node(counter)
			pos[counter] = (0, posCounter)
			labels[counter] = f"x{i+1}"
			inputNode.append(counter)
			posCounter += 1
			counter += 1

		posCounter = 1
		# 2. Nodes for hidden layer, if exist
		if (self.n_hidden_layer > 0):
			for i in range(self.n_hidden_layer):            
				# + the hidden layer bias
				G.add_node(counter, label="bias")    
				pos[counter] = (1+i, posCounter)
				labels[counter] = "1"
				biasNode.append(counter)
				posCounter += 1
				counter += 1
				# + hidden nodes
				for j in range(self.hidden_layer[i].n_neuron):
					G.add_node(counter)
					pos[counter] = (1+i, posCounter)
					labels[counter] = f"h{i+1}{j+1}"
					hiddenNode.append(counter)
					posCounter += 1
					counter += 1
				plt.annotate(f"{self.hidden_layer[i].activation_function.__name__}", xy=(1+i, posCounter-1), xytext=(1+i, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
							
		posCounter = 1
		# 3. Nodes for output layer
		# + output nodes
		for i in range(len(self.output[0])):
			G.add_node(counter)
			pos[counter] = (1+self.n_hidden_layer, posCounter)
			labels[counter] = f"o{i+1}"
			outputNode.append(counter)
			posCounter += 1
			counter += 1
		plt.annotate(f"{self.output_layer.activation_function.__name__}", xy=(1+self.n_hidden_layer, posCounter-1), xytext=(1+self.n_hidden_layer, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
			
		# nodes style
		options = {"edgecolors": "tab:gray", "node_size": 900, "alpha": 1}
		nx.draw_networkx_nodes(G, pos, nodelist=biasNode, node_color="tab:grey", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=inputNode, node_color="tab:red", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=hiddenNode, node_color="tab:blue", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=outputNode, node_color="tab:green", **options)
			
		#### 2.EDGES ADJUSTMENT
		edgeMap = []
		edge_labels = {}
		# 1. Edges for input layer to output layer (hidden is not exist)
		if (self.n_hidden_layer == 0):
			# connect input to output layer
			for i in range(self.input_size + 1):
				for j in range(len(self.output[0])):
					G.add_edge(i + 1, self.input_size + 2 + j)
					edgeMap.append((i + 1, self.input_size + 2 + j))
					if (i == 0):
						edge_labels[(i + 1, self.input_size + 2 + j)] = f"{self.output_layer.bias[j]:.2f}"
					else:
						edge_labels[(i + 1, self.input_size + 2 + j)] = f"{self.output_layer.weights[i-1][j]:.2f}"
		# 2. Edges for input layer to hidden layer if exist
		elif (self.n_hidden_layer > 0):
			# Connect input to hidden layer
			for i in range(self.input_size + 1):
				for j in range(self.hidden_layer[0].n_neuron):
					G.add_edge(i + 1, self.input_size + 2 + j + 1)
					edgeMap.append((i + 1, self.input_size + 2 + j + 1))
					if (i == 0):
						edge_labels[(i + 1, self.input_size + 2 + j + 1)] = f"{self.hidden_layer[0].bias[j]:.2f}"
					else:
						edge_labels[(i + 1, self.input_size + 2 + j + 1)] = f"{self.hidden_layer[0].weights[i-1][j]:.2f}"
			# Connect hidden layer to other hidden layer
			totalNeuronBeforeThisLayer = 1
			for i in range(self.n_hidden_layer - 1):
				for j in range(self.hidden_layer[i].n_neuron + 1):
					for k in range(self.hidden_layer[i+1].n_neuron):
						G.add_edge(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)
						edgeMap.append((self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1))
						if (j == 0):
							edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)] = f"{self.hidden_layer[i+1].bias[k]:.2f}"
						else:
							edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)] = f"{self.hidden_layer[i+1].weights[j-1][k]:.2f}"
				totalNeuronBeforeThisLayer = totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + 1
			# Connect last hidden layer to output layer
			for i in range(self.hidden_layer[-1].n_neuron + 1):
				for j in range(len(self.output[0])):
					G.add_edge(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)
					edgeMap.append((self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1))
					if (i == 0):
						edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)] = f"{self.output_layer.bias[j]:.2f}"
					else:
						edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)] = f"{self.output_layer.weights[i-1][j]:.2f}"
									
		# edges style
		edgeOptions = {"width": 2, "alpha": 0.7}
		nx.draw_networkx_edges(G, pos, edgelist=edgeMap, **edgeOptions)
			
		### 3.LABELS ADJUSTMENT
		# labels
		nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", font_color="whitesmoke")
		nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, label_pos=0.25, font_weight="bold", font_color="tab:gray")

		plt.tight_layout()
		plt.axis("off")
		plt.show()