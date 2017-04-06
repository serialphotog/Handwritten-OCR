import numpy as np 
import matplotlib.pyplot as plt

class GraphEngine:

	# The data points for our graph
	data_points = []

	# Adds a data point to our collection
	def add_plot_point(self, point):
		self.data_points.append(point)

	# Sets the title of our plot
	def set_title(self, title):
		plt.title(title)
		#self.figure.subtitle(title, fontsize=20)

	# Sets the axis labels of our plot
	def set_axis_labels(self, xlabel='Run', ylabel='Error Rate'):
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

	# Plots our data
	def plot(self):
		plt.plot([np.mean(self.data_points[i]) for i in range(len(self.data_points))])

	# Shows the data plot
	def show(self):
		self.plot
		plt.show()