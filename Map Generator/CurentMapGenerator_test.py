'''
To use this script,
fill config.txt with the following values (without numbers in brackets):
(0) File Manager Lib path
(1) Folder to save maps to

'''

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
	user_map_folder = fix_to_canonical_path(config[1])

except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")


from map_generation.CurentMapGenerator import CurentMapGenerator as generator

import numpy as np

def gaussian(x, mu = 0.0, sigma = 1.0):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def map_function(I1, I2):
	# generate signal grid
	x_space = np.arange(x_range[0], x_range[1], x_steps)
	y = np.zeros(x_space.size)

	# obtain point parameters
	coh = get_coh(I1, I2)
	width = get_width(I1, I2)

	# generate normal signal with parameters
	for i in range(x_space.size):
		y[i] = normal_with_parameters(x_space[i], coh, width)

	return x_space, y

def normal_with_parameters(x, coh, width):
	y = gaussian(x, sigma = (-width + 1))
	return y

	# spike = gaussian(x, sigma = 0.01 * (width + 1))
	# height = gaussian(x, sigma = (width + 1)) / (-coh + 0.01)
	# y = 0.0
	# if spike > height:
	# 	y = spike
	# else:
	# 	y = height
	# return y

	# return np.random.normal(x) + coh

def equal_pits_and_one_opt(I1, I2):
	minor_pits = [[4, 4], [4, 7], [7, 4], [7, 7]]
	opt_pit_idx = 2
	opt_pit_inner_r = 0.3
	outer_r = 1.2
	inner_r = 0.5

	resulting_field_value = 0.0
	for idx, pit in enumerate(minor_pits):
		r = np.linalg.norm(np.array([I1, I2]) - np.array(pit))
		if r > outer_r:
			r = float('inf')
		if idx is opt_pit_idx:
			if r < opt_pit_inner_r:
				r = opt_pit_inner_r
		else:
			if r < inner_r:
				r = inner_r
		resulting_field_value -= 1 / r

	return resulting_field_value
		
def sin_field(I1, I2):
	return np.sin(I1 * 8) * np.sin(I2 * 8)



get_coh = lambda I1, I2 : 1
get_width = equal_pits_and_one_opt
# get_width = sin_field

x_range = [-5, 5]
x_steps = 0.1

bounds = {
	'I1_min' : 3.0,
	'I1_max' : 8.0,
	'I2_min' : 3.0,
	'I2_max' : 8.0,
}

steps = {
	'I1' : 0.1,
	'I2' : 0.1
}

signal_type = 'ACF'


def test():
	print('testing')
	map_generator = generator(user_map_folder, 
		map_function, bounds, steps, signal_type)
	map_generator.generate_map()

test()