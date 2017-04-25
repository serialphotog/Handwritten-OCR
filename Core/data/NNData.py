import numpy as np 
import os

from Core.data.DataProcessor import DataProcessor

class NNData:

	##########
	# Loads and processes data from a local directory
	#
	# Params:
	#	* dir - Directory to load data from
	#
	# Returns the image data and correct values for those images
	##########
	def load_local_data(self, dir):
		# Ensure the data directory actually exists, else exit due to error
		if not os.path.isdir(dir):
			print "[ERROR]", dir, "is not a directory!"
			exit(-1)

		# Load the data
		files = self.__get_files_in_dir(dir)
		n_files = len(files)

		# Storage for processed image data and correct values
		image_data = np.zeros((n_files, 28*28)) # our NN works with 28px by 28px images
		correct_values = np.zeros((n_files, 10)) # Our NN recognizes digits 0-9 (or 10 values)

		# Process the data
		for i in range(n_files):
			img_file = os.path.join(dir, files[i])
			image_data[i] = DataProcessor.process_image_data(img_file)
			correct_values[i] = DataProcessor.get_correct_value(img_file)

		return image_data, correct_values

	##########
	# Returns a list of files in a directory
	# Params:
	#	dir - The directory to get file list for
	#
	# Returns the list of files in directory
	##########
	def __get_files_in_dir(self, dir):
		return [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

	##########
	# Extracts the real value from an array.
	# Our neural network outputs results, such as [13.57, 0, 2.3, ..., 6.7]. This method converts this into 
	# the real value it represents (0-9)
	##########
	@staticmethod
	def get_real_value(value_arr):
		highest = 0
		for i in range(len(value_arr)):
			if value_arr[i] > highest: 
				highest = i
		return highest