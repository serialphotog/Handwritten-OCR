import argparse

from Core.nn.Model import Model 
from Core.nn.NeuralNetwork import NeuralNetwork
from Core.data.NNData import NNData

def main():
	# Handle command line arguments
	parser = argparse.ArgumentParser(description="Neural Network Tester")
	parser.add_argument("-v", action="store_true", dest="verbose", 
		help="Enables verbose output.")
	parser.add_argument("-g", action="store_true", dest="enable_graph",
		help="Enables graph production.")
	args = parser.parse_args()

	# Build a new model
	model = Model.new_model([784, 500, 10])

	# Our data processor
	data_engine = NNData()

	# Start the testing
	neural_network = NeuralNetwork(model, epochs=10, verbose=args.verbose, 
		enable_graph=args.enable_graph)
	neural_network.train_mnist()
	#images, correct_vals = data_engine.load_local_data("/home/adam/projects/handwriting-samples/raw/")
	#Sneural_network.test(images, correct_vals)

if __name__ == '__main__':
	main()