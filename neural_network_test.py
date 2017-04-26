import argparse

from Core.data.NNData import NNData
from Core.nn.Model import Model 
from Core.nn.NeuralNetwork import NeuralNetwork


##########
# Class: App
#
# Represents the actual app itself
##########
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
		# Check if we specified a topology to run with
		if self.args.topology:
			topology = self.args.topology.split(',')
			for i in range(len(topology)): topology[i] = int(topology[i])
		else:
			topology = [784, 500, 10]

		# Check if we specified a number of epochs
		if self.args.epochs:
			epochs = int(self.args.epochs)
		else:
			epochs = 1

		# Check if we specified a title for the model
		if self.args.name:
			network_name = self.args.name 
		else:
			network_name = "Default Topology"

		# Build the network model and setup network
		model = Model.new_model(network_name, topology)
		neural_network = NeuralNetwork(model, epochs=epochs, verbose=self.args.verbose, 
			extreme_verbose=self.args.extreme_verbose, enable_graph=self.args.enable_graph)

		# Check if we specified data to train the network with
		if self.args.train_path:
			images, correct_vals = self.data_engine.load_local_data(self.args.train_path)
			neural_network.train(images, correct_vals)
		else:
			neural_network.train_mnist()

		# Check if we specified input data to test with network
		if self.args.test_path:
			images, correct_vals = self.data_engine.load_local_data(self.args.test_path)
			neural_network.test(images, correct_vals)


	##########
	# Parses the command line arguments
	##########
	def __parse_args(self):
		parser = argparse.ArgumentParser(description="Neural Network Tester")
		parser.add_argument("-v", action="store_true", dest="verbose", 
			help="Enables verbose output.")
		parser.add_argument("-ev", action="store_true", dest="extreme_verbose",
			help="Enables extreme verbose output")
		parser.add_argument("-g", action="store_true", dest="enable_graph",
			help="Enables graph production.")
		parser.add_argument("-t", dest="topology",
			help="Specifies a topology for the network (E.G. 784,500,10")
		parser.add_argument("-n", dest="name",
			help="Name for the network")
		parser.add_argument("-e", dest="epochs",
			help="Specifies the number of epochs")
		parser.add_argument("-train", dest="train_path",
			help="Specifies path to data to use for taining")
		parser.add_argument("-test", dest="test_path",
			help="Specifies a path for test data")
		self.args = parser.parse_args()


# Run everything
if __name__ == '__main__':
	app = App()
	app.run()