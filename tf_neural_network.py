import tensorflow as tf
from Util.input_data import read_data_sets # For MNIST dataset

from Util.GraphEngine import GraphEngine

class TFNeuralNetwork:

	TRAINING_BATCH_SIZE = 100

	##########
	# Initializes our tensorflow neural network
	# The layers parameter is a list representing our neural network:
	#		* The first entry represents the input layer
	#		* The last entry represents the output layer
	#		* All other entries represent hidden layers
	#		* The numbers for each entry represent number of nodes in the layer
	##########
	def __init__(self, layers, learning_rate=0.001, epochs=1, verbose=False, disable_graph=False):
		# Store cmd option flags
		self.verbose = verbose
		self.disable_graph = disable_graph

		# Setup error rate graph
		if not self.disable_graph:
			self.error_graph = GraphEngine()
			self.error_graph.set_title("Rate of Error")
			self.error_graph.set_axis_labels()

		# Setup the MNIST data
		self.MNIST = read_data_sets("MNIST_data/", one_hot=True)

		# The learning rate for the network
		self.learning_rate = learning_rate
		# Number of epochs for training cycle
		self.n_epochs = epochs

		# Build the network
		self.__build_network(layers)

	##########
	# Builds the neural network
	##########
	def __build_network(self, layers):
		# Build the layers
		self.__build_layers(layers)

		# Placeholders for data in the TF Graph
		self.graph_x = tf.placeholder("float", [None, self.n_inputs])
		self.graph_y = tf.placeholder("float", [None, self.n_outputs])

		# Initializes storage for weights and biases in our network
		self.__init_values()

		# Prediction engine and optimization
		self.prediction = self.network()
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.graph_y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		init = tf.global_variables_initializer()

		# Setup the session
		self.session = tf.Session()
		self.session.run(init)

	##########
	# The neural network
	##########
	def network(self):
		# Initialize the hidden layers
		for i in range(0, self.n_hidden_layers):
			if i == 0:
				# First hidden layer, start with input as input layer
				self.hidden_layers[i] = tf.add(tf.matmul(self.graph_x, self.weights['h0']), 
					self.biases['b0'])
			else:
				# Input is previous hidden layer
				self.hidden_layers[i] = tf.add(tf.matmul(self.hidden_layers[i-1], 
					self.weights[self.__get_translated_idx(i)]), self.biases[self.__get_translated_idx(i, 'b')])
			self.hidden_layers[i] = tf.nn.sigmoid(self.hidden_layers[i])

		# Output layer
		output_layer = tf.matmul(self.hidden_layers[self.n_hidden_layers-1], 
			self.weights['out']) + self.biases['out']

		return output_layer

	##########
	# Trains the neural network
	##########
	def train(self):
		# Train the network according to number of epochs
		for epoch in range(self.n_epochs):
			avg_cost = 0
			total_batch_size = int(self.MNIST.train.num_examples/self.TRAINING_BATCH_SIZE)

			# Loop over the batches
			for i in range(total_batch_size):
				batch_x, batch_y = self.MNIST.train.next_batch(self.TRAINING_BATCH_SIZE)

				# Run the optimizer
				_, error_rate = self.session.run([self.optimizer, self.cost], feed_dict={self.graph_x: batch_x, 
					self.graph_y: batch_y})
				avg_cost += error_rate/total_batch_size

				# Store error for graphing
				if not self.disable_graph:
					self.error_graph.add_plot_point(error_rate)

			# Print the epoch results to the terminal
			if self.verbose and epoch % 1 == 0: 
				print "Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost)

		print "Training completed!"

		# Calculate the network accuracy
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.graph_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# Output accuracy to terminal
		print "Accuracy:", accuracy.eval({self.graph_x: self.MNIST.test.images, 
			self.graph_y: self.MNIST.test.labels}, session=self.session)

		# Show the graphs
		if not self.disable_graph:
			self.error_graph.plot()
			self.error_graph.show()

	##########
	# Sets up network layers
	##########
	def __build_layers(self, layers):
		# The number of hidden layers
		self.n_hidden_layers = len(layers) - 2
		# Strip input and output layers, leaving hiden layers
		self.hidden_layer_nodes = layers[1:-1]
		# Storage for the hidden layers
		self.hidden_layers = [None] * self.n_hidden_layers
		# The number of input nodes
		self.n_inputs = layers[0]
		# The number of output nodes
		self.n_outputs = layers[len(layers) - 1]

	##########
	# Initializes the weights and biases for the network
	##########
	def __init_values(self):
		# Setup storage
		self.weights = {}
		self.biases = {}

		# Randomly assign weights and biases to start with
		for i in range(0, self.n_hidden_layers):
			# Weights
			if i == 0:
				# First hidden layer, start with input layer
				self.weights[self.__get_translated_idx(i)] = tf.Variable(tf.random_normal([self.n_inputs, 
					self.hidden_layer_nodes[0]]))
			else:
				# Continue with input being previous hidden layer
				self.weights[self.__get_translated_idx(i)] = tf.Variable(tf.random_normal([self.hidden_layer_nodes[i-1], 
					self.hidden_layer_nodes[i]]))

			# Biases
			self.biases[self.__get_translated_idx(i, 'b')] = tf.Variable(tf.random_normal([self.hidden_layer_nodes[i]]))

		# Add the outputs to weights and biases
		self.weights['out'] = tf.Variable(tf.random_normal([self.hidden_layer_nodes[self.n_hidden_layers-1], 
			self.n_outputs]))
		self.biases['out'] = tf.Variable(tf.random_normal([self.n_outputs]))

	##########
	# Returns index for weights and biases store
	##########
	def __get_translated_idx(self, idx, prefix='h'):
		return prefix + str(idx)