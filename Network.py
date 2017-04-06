import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

from GraphEngine import GraphEngine

class Network:
	# The loaded MNIST dataset
	MNIST_DATA = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# Each image is 28*28=784 pixels
	INPUT_LAYER_SIZE = 784

	# MNIST data represents digits 0-9 for total of 10 classes
	NUM_CLASSES = 10

	# The batch size for the training step
	TRAINING_BATCH_SIZE = 100

	# The learning rate for the network
	LEARNING_RATE = 0.001

	# The number of epochs for the network
	NUM_EPOCHS = 600

	# The placeholder for our data in the TensorFlow graph
	x = tf.placeholder("float", [None, INPUT_LAYER_SIZE])
	y = tf.placeholder("float", [None, NUM_CLASSES])

	def __init__(self, hidden_layers):
		# Set up our graph engine for data plotting
		self.graph_engine = GraphEngine()

		self.num_hidden_layers = len(hidden_layers)
		self.hidden_layers_config = hidden_layers
		self.hidden_layers = [None] * self.num_hidden_layers

		# Initialize weights and biases
		self.weights = {}
		self.biases = {}

		for i in range(0, self.num_hidden_layers):
			# Weights
			if i == 0:
				self.weights['h' + str(i)] = tf.Variable(tf.random_normal([self.INPUT_LAYER_SIZE, self.hidden_layers_config[0]]))
			else:
				self.weights['h' + str(i)] = tf.Variable(tf.random_normal([self.hidden_layers_config[i-1], self.hidden_layers_config[i]]))

			# Biases
			self.biases['b' + str(i)] = tf.Variable(tf.random_normal([self.hidden_layers_config[i]]))

		# Add outputs to weights and biases
		self.weights['out'] = tf.Variable(tf.random_normal([self.hidden_layers_config[self.num_hidden_layers-1], self.NUM_CLASSES]))
		self.biases['out'] = tf.Variable(tf.random_normal([self.NUM_CLASSES]))


	def net(self):
		# Initialize the hidden layers
		for i in range(0, self.num_hidden_layers):
			if i == 0:
				self.hidden_layers[i] = tf.add(tf.matmul(self.x, self.weights['h0']), self.biases['b0'])
			else:
				self.hidden_layers[i] = tf.add(tf.matmul(self.hidden_layers[i-1], self.weights['h' + str(i)]), self.biases['b' + str(i)])

			self.hidden_layers[i] = tf.nn.sigmoid(self.hidden_layers[i])	

		# Output layer
		output_layer = tf.matmul(self.hidden_layers[self.num_hidden_layers-1], self.weights['out']) + self.biases['out']

		return output_layer	

	def train(self):
		prediction = self.net()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(cost)
		init = tf.global_variables_initializer()

		self.costs = []

		with tf.Session() as session:
			session.run(init)

			# Train the network
			for epoch in range(self.NUM_EPOCHS):
				avg_cost = 0
				total_batch = int(self.MNIST_DATA.train.num_examples/self.TRAINING_BATCH_SIZE)

				# Loop over the batches
				for i in range(total_batch):
					batch_x, batch_y = self.MNIST_DATA.train.next_batch(self.TRAINING_BATCH_SIZE)

					# Run the optimization
					_, c = session.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
					self.graph_engine.add_plot_point(c)
					avg_cost += c/total_batch

				# Print epoch results to termianl
				if epoch % 1 == 0:
					print "Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost)

			print "Training completed!"

			# Calculate accuracy
			correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print "Accuracy:", accuracy.eval({self.x: self.MNIST_DATA.test.images, self.y: self.MNIST_DATA.test.labels})


network = Network([250, 250])
network.train()
network.graph_engine.plot()
network.graph_engine.set_title("Rate of Error")
network.graph_engine.set_axis_labels()
network.graph_engine.show()
