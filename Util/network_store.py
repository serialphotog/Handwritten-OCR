import os
import pickle
import random
import string
import zipfile

class NetworkStore:

	##########
	# Initializes our network storage manager
	##########
	def __init__(self, path):
		# Generate working dir (Will only work on *nix systems)
		folder = self.__generate_random_string()
		self.working_dir = os.path.join('/tmp', folder)

	##########
	# Returns a randomly generate string on length n (used for creating temporary folders)
	##########
	def __generate_random_string(self, n=15):
		return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

	##########
	# Loads a saved network model
	##########
	def load(self, save_path):
		if os.path.isfile(save_path):
			self.__extract_save(path)

	##########
	# Creates a save file
	##########
	def save(self, name, location):
		# Create the save archive
		zip_handle = zipfile.ZipFile(os.path.join(location, name+'.zip'), 'w', zipfile.ZIP_DEFLATED)
		self.__zip_directory(self.working_dir)
		zip_handle.close()

	##########
	# Pickles an object and saves it to disk
	##########
	def save_object(self, obj, name):
		# The path to the pickle file
		path = os.path.join(self.working_dir, name + '.pckl')

		# Check if pickle exists
		if os.path.isfile(path):
			# If so, delete it
			os.remove(path)

		# Pickle the object
		path = open(path, 'wb')
		pickle.dump(obj, path)
		path.close()

	##########
	# Loads an object from a pickled file
	##########
	def load_object(self, name):
		# The path to the pickle file
		path = os.path.join(self.working_dir, name + '.pckl')

		# Load the object
		if os.path.isfile(path):
			path = open(path, 'r')
			obj = pickle.load(path)
			path.close()
			return obj
		return None


	##########
	# Extracts the save data to /tmp on the system (This will only work on *nix systems)
	##########
	def __extract_save(self, path):
		# Generate random folder name
		folder = self.__generate_random_string()
		self.dir = os.path.join('/tmp', folder)

		print "[INFO] Extracting save to ", self.dir

		# Extract the save files
		with zipfile.ZipFile(path, "r") as zip:
			zip.extractall(self.dir)

	##########
	# Compresses an entire directory into a zipfile
	##########
	def __zip_directory(self, dir, zip_handle):
		for root, directories, files in os.walk(path):
			for file in files:
				zip_handle.write(os.path.join(root, file))