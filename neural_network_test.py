import argparse

from Core.nn.Model import Model 
from Core.nn.NeuralNetwork import NeuralNetwork
from Core.data.NNData import NNData

class App:
	##########
	# Initialize application components
	##########
	def __init__(self):
		# Parse command line arguments
		self.__parse_args()

		# Initialize the data engine
		self.data_engine = NNData()

	##########
	# Runs the app
	##########
	def run(self):
		model = Model.new_model([784, 500, 10])

		# Start the testing
		neural_network = NeuralNetwork(model, epochs=10, verbose=self.args.verbose, 
			enable_graph=self.args.enable_graph)
		neural_network.train_mnist()
		#images, correct_vals = data_engine.load_local_data("/home/adam/projects/handwriting-samples/raw/")
		#Sneural_network.test(images, correct_vals)


	##########
	# Parses the command line arguments
	##########
	def __parse_args(self):
		parser = argparse.ArgumentParser(description="Neural Network Tester")
		parser.add_argument("-v", action="store_true", dest="verbose", 
			help="Enables verbose output.")
		parser.add_argument("-g", action="store_true", dest="enable_graph",
			help="Enables graph production.")
		self.args = parser.parse_args()


# Run everything
if __name__ == '__main__':
	app = App()
	app.run()