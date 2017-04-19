import cv2
import numpy as np 
import os

from scipy import ndimage

class DataProcessor:

	##########
	# Static method to process the image data files into a form that's more efficent for our neural
	# network to work with.
	# 
	# Params:
	#	* img - The path to the image to process
	##########
	@staticmethod
	def process_image_data(img):
		# Read in, invert and resize the image
		inverted_img = cv2.imread(img, 0)
		inverted_img = cv2.resize(255-inverted_img, (28, 28)) # Inverts and resizes image

		# Generate white text on black background (Improves the inverted image)
		(threshold, inverted_img) = cv2.threshold(inverted_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Centrally align the image (as best as possible)
		inverted_img = DataProcessor.shift_image(inverted_img)

		# Our neural network expects pixel values to be in the range of 0-1, but a standard image
		# has pixel values in the range 0-255, so we need to flatten the image
		flattened_img = inverted_img.flatten() / 255.0

		return flattened_img

	##########
	# Static method to shift an image so it is more centrally aligned.
	# The openCV documentation is very helpful for figuring this out:
	#	http://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html#gsc.tab=0
	#
	# Params:
	#	* img - The image to shift (matrix of pixel values)
	##########
	@staticmethod
	def shift_image(img):
		# Calculate the shift amounts
		cy, cx = ndimage.measurements.center_of_mass(img)
		rows, cols = img.shape
		shiftx = np.round(cols/2.0-cx).astype(int)
		shifty = np.round(rows/2.0-cy).astype(int)

		# Build the transformation matrix and perform the shift operation
		transform = np.float32([[1, 0, shiftx], [0, 1, shifty]])
		shifted_img = cv2.warpAffine(img, transform, (cols, rows))

		return shifted_img

	##########
	# Static method to determin the correct value for an image.
	#
	# Params:
	#	* img - The image to load value for
	##########
	@staticmethod
	def get_correct_value(img):
		# Storage matrix for correct value
		correct_val = np.zeros(10)

		# The correct value is encoded in filename
		filename = os.path.basename(img)
		num = filename[0] # The correct value is the first character of the filename

		# Store the correct value
		correct_val[int(num)] = 1

		return correct_val