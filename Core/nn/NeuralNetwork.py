import tensorflow as tf

from Core.util.input_data import read_data_sets # For MNIST dataset
from Core.util.Grapher import Grapher

class NeuralNetwork:

	TRAINING_BATCH_SIZE = 100

	##########
	# Initializes our tensorflow neural network
	##########
	def __init__(self, model, learning_rate=0.001, epochs=1, verbose=False, extreme_verbose=False, enable_graph=False):
		# The model
		self.model = model

		# Store cmd option flags
		self.verbose = verbose
		self.extreme_verbose = extreme_verbose
		self.enable_graph = enable_graph

		# Setup error rate graph
		if self.enable_graph:
			self.error_graph = Grapher()
			self.error_graph.set_title("Rate of Error: " + model.name)
			self.error_graph.set_axis_labels()

		# Setup the MNIST data
		self.MNIST = read_data_sets("MNIST_data/", one_hot=True)

		# The learning rate for the network
		self.learning_rate = learning_rate
		# Number of epochs for training cycle
		self.n_epochs = epochs

		# Build the network
		self.__build_network()

	##########
	# Builds the neural network
	##########
	def __build_network(self):
		# Placeholders for data in the TF Graph
		self.graph_x = tf.placeholder("float", [None, self.model.n_inputs])
		self.graph_y = tf.placeholder("float", [None, self.model.n_outputs])

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
		for i in range(0, self.model.n_hidden_layers):
			if i == 0:
				# First hidden layer, start with input as input layer
				self.model.hidden_layers[i] = tf.add(tf.matmul(self.graph_x, self.model.weights['h0']), 
					self.model.biases['b0'])
			else:
				# Input is previous hidden layer
				self.model.hidden_layers[i] = tf.add(tf.matmul(self.model.hidden_layers[i-1], 
					self.model.weights[Model.get_translated_idx(i)]), self.model.biases[Model.get_translated_idx(i, 'b')])
			self.model.hidden_layers[i] = tf.nn.sigmoid(self.model.hidden_layers[i])

		# Output layer
		output_layer = tf.matmul(self.model.hidden_layers[self.model.n_hidden_layers-1], 
			self.model.weights['out']) + self.model.biases['out']

		return output_layer

	##########
	# Trains the neural network
	#
	# Params:
	#	images - images to train on (default = None)
	#	labels - correct values (default = None)
	#	n_batchs - Number of batches to run (default = 0)
	#
	# Note that if n_batches = 0 we default to using mnist data
	##########
	def __train(self, images=None, labels=None, n_batches=0):
		print "Running training sequence"

		# Run through n epochs
		for epoch in range(self.n_epochs):
			avg_cost = 0
			n_images = n_batches if n_batches > 0 else len(images)

			# Train through all images
			for i in range(n_images):
				# Get batch data
				if n_batches == 0:
					batch_x = images
					batch_y = labels 
				else:
					batch_x, batch_y = self.MNIST.train.next_batch(self.TRAINING_BATCH_SIZE)

				# Run the training step with the optimizer
				_, error_rate = self.session.run([self.optimizer, self.cost], feed_dict={self.graph_x: batch_x,
					self.graph_y: batch_y})

				# Calculate the cost
				avg_cost += error_rate / n_images

				# Store the error rate for graphing
				if self.enable_graph:
					self.error_graph.add_plot_point(error_rate)

			# Print the epoch results, if verbose output is enables
			if self.verbose and epoch % 1 == 0:
				print "Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost)

		print "Training step completed!"

		# Calculate the accuracy
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.graph_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# Output the accuracy results
		print "Accuracy:", self.accuracy.eval({self.graph_x: self.MNIST.test.images, 
			self.graph_y: self.MNIST.test.labels}, session=self.session)

		# Show the graphs
		if self.enable_graph:
			self.error_graph.plot()
			self.error_graph.show()

	##########
	# Trains the neural network using a generic dataset
	##########
	def train(self, images, labels):
		self.__train(images, labels, 0)

	##########
	# Trains the neural network using the MNIST dataset
	##########
	def train_mnist(self):
		# Calculate the number of batches
		n_batches = int(self.MNIST.train.num_examples / self.TRAINING_BATCH_SIZE)
		# Run the training
		self.__train(n_batches=n_batches)

	def test(self, image_data, correct_vals):
		print "Running test case: "
		_, error_rate = self.session.run([self.prediction, self.cost], feed_dict={self.graph_x: image_data, self.graph_y: correct_vals})
		print "Error rate: ", error_rate
		print "Accuracy: ", self.session.run(self.accuracy, feed_dict={self.graph_x: image_data, self.graph_y: correct_vals})
