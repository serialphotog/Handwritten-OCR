import cv2
import math
import numpy as np 
import os

# Square size to resize images to
IMAGE_RESIZE = 28

# Returns image data array and correct values array
def get_sample_image_data(path):
	# Get data from directory
	files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
	n_files = len(files)

	# Image data and correct values storage
	image_data = np.zeros((n_files, IMAGE_RESIZE*IMAGE_RESIZE))
	correct_vals = np.zeros((n_files, 10))

	# Loop over images
	for i in range(n_files):
		# Read, invert and resize the image
		gray_scale = cv2.imread(path+files[i], 0)
		gray_scale = cv2.resize(255-gray_scale, (IMAGE_RESIZE, IMAGE_RESIZE))

		# We want a white image on a black background
		(threshold, gray_scale) = cv2.threshold(gray_scale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Flatten the image:
		# Our neural network expects that our pixel values be in the range 0-1, not 0-255
		flattened = gray_scale.flatten() / 255.0

		# Store the image and generate correct value
		image_data[i] = flattened
		correct_val = np.zeros(10)

		filename = os.path.basename(files[i]) # Get the name of file
		num = filename[0] # Get correct value from filename
		correct_val[int(num)] = 1 # Set the correct val

		cv2.imwrite("/home/adam/test/"+filename[0]+"_"+str(i)+".jpg", gray_scale)

		# Store correct value matrix
		correct_vals[i] = correct_val

	return image_data, correct_vals