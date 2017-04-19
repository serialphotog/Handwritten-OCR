import argparse

from Core.nn.Model import Model 
from Core.nn.NeuralNetwork import NeuralNetwork
from Core.util.handwriting_sample import get_sample_image_data

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

	# Start the testing
	neural_network = NeuralNetwork(model, epochs=10, verbose=args.verbose, 
		enable_graph=args.enable_graph)
	neural_network.train()
	# images, correct_vals = get_sample_image_data("/home/adam/projects/handwriting-samples/raw/")
	# neural_network.test(images, correct_vals)

if __name__ == '__main__':
	main()