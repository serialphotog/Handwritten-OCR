import tensorflow as tf 
import Util.input_data as input_data

class NeuralNetwork:

	# Initializes the neural network
	def __init__(self):
		# The MNIST dataset
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		# Place holder for the image data
		self.x = tf.placeholder("float", [None, 784])

		# Weights and biases
		W = tf.Variable(tf.zeros([784, 10]))
		b = tf.Variable(tf.zeros([10]))

		self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

		self.y_ = tf.placeholder("float", [None, 10])

		cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

		self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

		init = tf.initialize_all_variables()

		# Create the session
		self.session = tf.Session()
		self.session.run(init)

	# Trains our neural network
	def train(self):
		# Use 1000 batches of 100 to train the network
		for i in range(1000):
			batch_xs, batch_ys = self.mnist.train.next_batch(100)

			# Run the training step
			self.session.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print self.session.run(accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels})

nn = NeuralNetwork()
nn.train()