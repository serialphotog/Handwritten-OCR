import tensorflow as tf 

class Model:

	##########
	# Initializes a model
	# The layers parameter is a list representing our neural network:
	#		* The first entry represents the input layer
	#		* The last entry represents the output layer
	#		* All other entries represent hidden layers
	#		* The numbers for each entry represent number of nodes in the layer
	##########
	def __init__(self, name, layers, weights, biases):
		self.name = name
		self.layers = layers 
		self.weights = weights
		self.biases = biases 

		self.__build_model()

	##########
	# Builds the model
	##########
	def __build_model(self):
		# The number of hidden layers
		self.n_hidden_layers = len(self.layers) - 2
		# Strip the input and output layers, just leaving hidden layers
		self.hidden_layer_nodes = self.layers[1:-1]
		# Storage for hidden layers
		self.hidden_layers = [None] * self.n_hidden_layers
		# The number of input nodes 
		self.n_inputs = self.layers[0]
		# The number of output nodes
		self.n_outputs = self.layers[len(self.layers) - 1]

	##########
	# Static method to build a new, untrained model
	##########
	@staticmethod
	def new_model(name, layers):
		# The number of hidden layers
		n_hidden_layers = len(layers) - 2
		# Strip the input and output layers, just leaving hidden layers
		hidden_layer_nodes = layers[1:-1]
		# Storage for hidden layers
		hidden_layers = [None] * n_hidden_layers
		# The number of input nodes 
		n_inputs = layers[0]
		# The number of output nodes
		n_outputs = layers[len(layers) - 1]

		# build the weights and biases
		weights = {}
		biases = {}

		for i in range(n_hidden_layers):
			# Weights 
			if i == 0:
				# First hidden layer, input is input layer
				weights[Model.get_translated_idx(i)] = tf.Variable(tf.random_normal([n_inputs,
					hidden_layer_nodes[0]]))
			else:
				# Input is previous hidden layer
				weights[Model.get_translated_idx(i)] = tf.Variable(tf.random_normal([hidden_layer_nodes[i-1],
					hidden_layer_nodes[i]]))

			# Biases
			biases[Model.get_translated_idx(i, 'b')] = tf.Variable(tf.random_normal([hidden_layer_nodes[i]]))

		# Add outputs to weights and biases
		weights['out'] = tf.Variable(tf.random_normal([hidden_layer_nodes[n_hidden_layers-1],
			n_outputs]))
		biases['out'] = tf.Variable(tf.random_normal([n_outputs]))

		# Return the new model
		return Model(name, layers, weights, biases)

	##########
	# Static method to translate index for weights and biases
	##########
	@staticmethod
	def get_translated_idx(idx, prefix='h'):
		return prefix + str(idx)