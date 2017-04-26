import tensorflow as tf

from Core.util.input_data import read_data_sets # For MNIST dataset

from Core.data.NNData import NNData as Data
from Core.nn.Model import Model
from Core.util.Grapher import Grapher
from Core.util.Timer import Timer 

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
		self.__print_header("Running training sequence")

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
		timer = Timer()
		timer.start()
		self.__train(images, labels, 0)
		print "Training completed in:", timer.stop(), "seconds"

	##########
	# Trains the neural network using the MNIST dataset
	##########
	def train_mnist(self):
		timer = Timer()
		timer.start()
		# Calculate the number of batches
		n_batches = int(self.MNIST.train.num_examples / self.TRAINING_BATCH_SIZE)
		# Run the training
		self.__train(n_batches=n_batches)
		print "Training completed in:", timer.stop(), "seconds"

	##########
	# Tests images agains the neural network
	#
	# Params:
	# 	* image_data - The image data (pixel values stored in list)
	#	* correct_vals - Correct values for images
	##########
	def test(self, image_data, correct_vals):
		self.__print_header("Running test case")

		prediction = [tf.reduce_max(self.graph_y), tf.argmax(self.graph_y, 1)[0]]
		accuracy = 0

		for i in range(len(image_data)):
			# Run through the network
			guess = self.session.run(self.prediction, feed_dict={self.graph_x: [image_data[i]]})
			guess_value = Data.get_real_value(guess[0])
			correct_value = Data.get_real_value(correct_vals[i])

			if guess_value == correct_value:
				accuracy += 1 # Correct guess

			if self.extreme_verbose:
				print "Predicted:", Data.get_real_value(guess[0]), "Correct:", Data.get_real_value(correct_vals[i])

		# Calculate the accuracy
		accuracy = (float(accuracy) / len(image_data)) * 100
		print "Accuracy:", "%.2f%%" % (accuracy)

	##########
	# Simple function to print a nice-looking header to console
	##########
	def __print_header(self, header):
		print "*"*50
		print "\t", header
		print "*"*50
