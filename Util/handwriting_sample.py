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

		#gray_scale = fit_image(gray_scale)

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

# Fits our image data into a 20x20 area
#
# The MNIST samples all fit within a 20x20 area of the total 28x28 image. To account for this, we fit 
# our samples into a 20x20 area and then ensure that our overall image is the full 28x28
# Returns the new image data
def fit_image(image):
	# Fit image into 20x20 by removing every completely black row and column 
	while np.sum(image[0]) == 0:
		image = image[1:]
	while np.sum(image[:,0]) == 0:
		image = np.delete[image, 0, 1]
	while np.sum(image[-1]) == 0:
		image = image[:-1]
	while np.sum(image[:,-1]) == 0:
		image = np.delete(image, -1, 1)

	# Resize to fit the 20x20
	rows, columns = image.shape
	if rows > columns:
		scale_factor = 20.0/rows
		rows = 20
		columns = int(round(columns * scale_factor))
		image = cv2.resize(image, (columns, rows))
	else:
		scale_factor = 20.0/columns
		columns = 20
		rows = int(round(rows * scale_factor))
		image = cv2.resize(image, (columns, rows))

	# Resize total image to be full 28x28
	colPadding = (int(math.ceil((IMAGE_RESIZE-columns)/2.0)), int(math.floor((IMAGE_RESIZE-columns)/2.0)))
	rowPadding = (int(math.ceil((IMAGE_RESIZE-rows)/2.0)), int(math.floor((IMAGE_RESIZE-rows)/2.0)))
	image = np.lib.pad(image, (rowPadding, colPadding), 'constant')

	return image