import sys

# LOADING CONFIG
def load_config():
	with open('config.cfg') as config:
		return [line.strip() for line in config.readlines()]

def fix_to_canonical_path(path):
	canonical_path = path
	if not path[-1] is '/':
		canonical_path += '/'
	return canonical_path

try:
	config = load_config()
	file_manager_lib_path = fix_to_canonical_path(config[0])
except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")

# Path to File Manager Lib
sys.path.insert(0, file_manager_lib_path)

from file_managers.SignalFileManager import SignalFileManager as file_manager

import numpy as np

class CurentMapGenerator():
	def __init__(self, saving_folder, 
			map_function, bounds, steps, signal_type = 'ACF'):
		self.map_function = map_function
		self.bounds = bounds
		self.steps = steps
		self.signal_type = signal_type
		self.file_manager = file_manager(saving_folder, 'generated', 
			experiment_folder_name = 'gauss_up_left')
		print(self.__class__.__name__, 'inited!')

	def generate_map(self):
		I1_space = self.get_arange('I1')
		I2_space = self.get_arange('I2')

		for I1 in I1_space:
			for I2 in I2_space:
				x, y = self.generate_signal(I1, I2)
				self.file_manager.save_variable(self.signal_type, I1, I2, x, y)

	def get_arange(self, variable):
		return np.arange(self.bounds[variable + '_min'], self.bounds[variable + '_max'], 
			self.steps[variable])

	def generate_signal(self, I1, I2):
		return self.map_function(I1, I2)
