import argparse

from neural_network import NeuralNetwork 
from Util.handwriting_sample import get_sample_image_data

def main():
	# Handle command line arguments
	parser = argparse.ArgumentParser(description="Neural Network Tester")
	parser.add_argument("-v", action="store_true", dest="verbose", 
		help="Enables verbose output.")
	parser.add_argument("-g", action="store_true", dest="disable_graph",
		help="Disables graph production.")
	args = parser.parse_args()

	# Start the testing
	neural_network = NeuralNetwork([784, 200, 10], epochs=5, verbose=args.verbose, 
		disable_graph=args.disable_graph)
	neural_network.train()
	images, correct_vals = get_sample_image_data("/home/adam/projects/handwriting-samples/raw/")
	neural_network.test(images, correct_vals)

if __name__ == '__main__':
	main()