import time

class Timer:

	##########
	# Starts the timer
	##########
	def start(self):
		self.time_start = time.time()

	##########
	# Stops the timer and returns the elapsed time
	##########
	def stop(self):
		self.time_stop = time.time()
		self.elapsed = self.time_stop - self.time_start
		return self.elapsed

	##########
	# Returns the elapsed time 
	##########
	def get_elapsed(self):
		return self.elapsed