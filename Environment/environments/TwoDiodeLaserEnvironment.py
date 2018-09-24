import sys
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import decimal

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# LOADING CONFIG
def load_config():
	with open('config.cfg') as config:
		return [line.strip() for line in config.readlines()]

try:
	config = load_config()
	measurement_lib_path = config[0]
	serial_laser_control_lib_path = config[1]
	signal_processing_lib_path = config[2]
	file_manager_lib_path = config[3]
	first_laser_diode_serial_port = config[4]
	second_laser_diode_serial_port = config[5]
	premade_measurement_folder_path = config[9]
except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")

# Path to Measurement Lib
sys.path.insert(0, measurement_lib_path)
# Path to Serial Laser Control Lib
sys.path.insert(0, serial_laser_control_lib_path)
# Path to Signal Processing Lib
sys.path.insert(0, signal_processing_lib_path)
# Path to File Manager Lib
sys.path.insert(0, file_manager_lib_path)

from devices.ACFDevice import ACFDevice
#from devices.RFDevice import RFDevice
from devices.OSDevice import OSDevice
from devices.FileVirtualDevice import FileVirtualDevice

from lazer_serial_controllers.LazerSerialControllerSingleLD import LazerSerialControllerSingleLD
from lazer_serial_controllers.EmptyVirtualController import EmptyVirtualController

from signal_processors.ACFSignalProcessor import ACFSignalProcessor
from signal_processors.RFSignalProcessor import RFSignalProcessor
from signal_processors.OSSignalProcessor import OSSignalProcessor

from file_managers.SignalFileManager import SignalFileManager


from gym import spaces
from gym import Env


class TwoDiodeLaserEnvironment(Env):
	def __init__(self, saving_path, if_save_pics = False, if_save_param_logs = False, if_save_csv = False, if_penalize = True):
		self.saving_path = saving_path
		self.if_save_pics = if_save_pics
		self.if_save_param_logs = if_save_param_logs
		self.if_save_csv = if_save_csv

		self.num_actions = 5
		self.state_size = 2
		self.is_infinite = True
		# self.needs_reset = False
		# self.is_always_done = True
		self.if_penalize = if_penalize

		self.step_value = 0.1
		self.step_time = 1.0
		self.is_scanning_current_poll_time = 0.05
		self.supported_variables = ['I1', 'I2']
		self.actions = [['I1', 'up'], ['I1', 'down'], ['I2', 'up'], ['I2', 'down']]
		self.constraints = {
		'I1_min' : 0.5,
		'I1_max' : 6.0,
		'I2_min' : 0.5,
		'I2_max' : 6.0
		}		
		self.values = {
		'I1' : None,
		'I2' : None
		}


		# self.acf_dev = ACFDevice()
		# # self.rf_dev = RFDevice()
		# self.os_dev = OSDevice()

		# Constraints you specified would be replaced by the bound values of available files
		self.acf_dev = FileVirtualDevice('ACF', self.constraints, self.values, 
			premade_measurement_folder_path, file_type = 'csv', delimiter = ',')
		# self.rf_dev = FileVirtualDevice('RF', self.constraints, self.values, premade_measurement_folder_path)
		# self.os_dev = FileVirtualDevice('OS', self.constraints, self.values, premade_measurement_folder_path, [None, 1080])

		# self.controllers = {
		# 'I1' : LazerSerialControllerSingleLD(first_laser_diode_serial_port),
		# 'I2' : LazerSerialControllerSingleLD(second_laser_diode_serial_port)
		# }

		self.controllers = {
		'I1' : EmptyVirtualController(),
		'I2' : EmptyVirtualController()
		}

		self.step_precision = -decimal.Decimal(str(self.step_value)).as_tuple().exponent

		
		self.acf_sp = ACFSignalProcessor()
		# self.rf_sp = RFSignalProcessor()
		# self.os_sp = OSSignalProcessor()
		self.file_manager = SignalFileManager(saving_path + '/csv/', 'nn_steps')

		self.acf_fwhm = 0.0
		self.acf_coh = 0.0
		self.os_fwhm = 0.0

		self.steps = 0
		self.done_episodes = 0

		self.init_currents()
		# # print('TwoDiodeLaserEnvironment', 'inited!')

		low = np.array([self.constraints['I1_min'], self.constraints['I2_min'], 2.4])
		high = np.array([self.constraints['I1_max'], self.constraints['I2_max'], 9.9])

		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(low=low, high=high)


		self.acf_fwhm_log = open("acf_fwhm_log.txt","w+")
		# self.acf_coh_log = open("acf_coh_log.txt","w+")

	def close(self):
		pass

	def init_currents(self):
		for variable in self.supported_variables:
			self.values[variable] = float(self.controllers[variable].get_current())
			# # print(self.values[variable])

	def check_variable(self, variable):
		is_supported = True
		if not variable in self.supported_variables:
			# # print("Variable '" + str(variable) + "'", 'is not supported')
			is_supported = False
		return is_supported

	def restrict_value(self, variable, value):
		result = value
		try:
			min_value = self.constraints[str(variable) + '_min']
			if value < min_value:
				# # print('Restricting', variable, 'variable to', min_value)
				result = min_value
			max_value = self.constraints[str(variable) + '_max']
			if value > max_value:
				# # print('Restricting', variable, 'variable to', max_value)
				result = max_value
		except:
			pass
		return result

	def int_str_to_bool(self, string):
		answer = None
		try:
			answer = bool(int(string))
		except:
			# print('String:', string, 'is not convertable to int')
			pass
		return answer

	def set_variable(self, variable, value):
		if self.check_variable(variable):
			value = self.restrict_value(variable, value)
			self.set_current(variable, value)

	def set_current(self, current, value):
		self.controllers[current].scan_current(value, self.step_time)
		self.values[current] = value
		self.wait_for_scan()

	def reset(self):
		# # print('Current enviroment values:', self.values)
		self.set_random_var_values()
		# # print('Enviroment reset to:', self.values)
		return self.get_state()

	def set_random_var_values(self):
		for var in self.values:
			self.values[var] = self.get_var_valid_random_value(var)

	def get_var_valid_random_value(self, variable):
		while True:
			random_var_value = np.random.uniform(self.constraints[variable + '_min'], 
				self.constraints[variable + '_max'])
			random_var_value = round(random_var_value, self.step_precision)	
			if self.if_var_value_allowed(variable, random_var_value):
				break		
		return random_var_value		

	def if_var_value_allowed(self, variable, value):
		is_allowed = False
		if self.constraints[variable + '_min'] <= value <= self.constraints[variable + '_max']:
			is_allowed = True
		return is_allowed

	def step(self, action):
		# print(action)
		self.steps += 1
		penalty = self.peform_action(action)
		reward, if_done = self.check_goal()
		state = self.get_state()
		# return state, (reward + penalty), if_done, {}
		return state, (reward + penalty), if_done

	def peform_action(self, action):
		'''
		0: Stay
		1: I1_up
		2: I1_down
		3: I2_up
		4: I2_down
		'''
		if_changes_anything = True
		if not action is 0:
			if_changes_anything = self.change_current(self.actions[action - 1][0], self.actions[action - 1][1])
		else:
			# # print('Staying on the same place')	
			pass
		penalty = 0.0		
		if self.if_penalize:
			if not if_changes_anything:
				penalty = -1.0
		# print('penalty', penalty)
		return penalty

	def change_current(self, current, direction):
		if direction is 'up':
			value = self.values[current] + self.step_value
		if direction is 'down':
			value = self.values[current] - self.step_value

		# # print('Setting', current, 'to', value)

		self.set_variable(current, value)

		return self.if_value_in_constraints(current, value)

	def if_value_in_constraints(self, variable, value):
		eps = 0.00001
		if_value_in_constraints = True
		if value > self.constraints[variable + '_max'] + eps:
			if_value_in_constraints = False
		if value < self.constraints[variable + '_min'] - eps:
			if_value_in_constraints = False
		return if_value_in_constraints

	def wait_for_scan(self):
		while True:
			if not self.controllers['I1'].is_scanning_current() and not self.controllers['I2'].is_scanning_current():
				break
			time.sleep(self.is_scanning_current_poll_time)

	def check_goal(self):
		if_done = False
		# if self.is_always_done is True:
		# 	if_done = True
		# else:
		# 	if_done = False

		acf_x, acf_y = self.acf_dev.get_acf()
		# os_x, os_y = self.os_dev.get_os()
		# rf_x, rf_y = self.rf_dev.get_rf()

		if self.if_save_pics:
			self.save_pic(acf_x, acf_y, 'acf')
			# self.save_pic(os_x, os_y, 'os')
			# self.save_pic(rf_x, rf_x, 'rf')
			
		if self.if_save_csv:
			self.file_manager.save_acf(self.values['I1'], self.values['I2'], acf_x, acf_y)
			# self.file_manager.save_os(self.values['I1'], self.values['I2'], os_x, os_y)

		try:
			acf_fwhm = self.acf_sp.get_acf_env_fwhm(time=acf_x, acf=acf_y)
			# acf_coh = self.acf_sp.get_acf_coh(time=acf_x, acf=acf_y)
			# os_fwhm = self.os_sp.get_wl_rms(os_x,os_y)
			# rf_med = self.rf_sp.get_median_amplitude(rf_y)
		except:
			# # print('Data obtaining failed, setting reward to 0.0')
			reward = 0.0
			return reward, if_done

		# print(self.values)
		# print('acf_fwhm', acf_fwhm)
		# # # print('acf_coh', acf_coh)
		# # # print('os_fwhm', os_fwhm)
		# # # print('rf_med', rf_med)

		# reward = self.get_reward_0(acf_coh, acf_fwhm, os_fwhm)
		# reward = self.get_reward_1(acf_coh, acf_fwhm, os_fwhm, rf_med)
		reward = self.get_reward_2(acf_fwhm)
		if_done = self.if_done(reward)
		if if_done:
			print('DONE')
		# print('reward:', reward, end='\r')

		if self.if_save_param_logs:
			self.acf_fwhm_log.write(str(self.steps) + ' ' + str(self.acf_fwhm) + '\n')
			# self.acf_coh_log.write(str(self.steps) + ' ' + str(self.acf_coh) + '\n')

		self.acf_fwhm = acf_fwhm
		# self.acf_coh = acf_coh
		# self.os_fwhm = os_fwhm	

		return reward, if_done

	def if_done(self, reward):
		if_done = False
		if reward >= 1:
		# if reward >= 1 or self.steps > 300:
			if_done = True
			self.steps = 0
			self.done_episodes += 1
			print(self.done_episodes)
		return if_done

	def get_state(self):
		# return [self.values[variable] for variable in self.supported_variables]
		currents = [self.values[variable] for variable in self.supported_variables]
		# currents.append(self.acf_fwhm)
		return currents

	def save_pic(self, x, y, name):
		plt.close()
		plt.plot(x, y)
		plt.savefig(self.saving_path +  '/pics/'+name+'/'
			+ str(self.steps) + '_'
			+ "%.1f" % self.values['I1'] + '_' + "%.1f" % self.values['I2'] + '.png')

	def get_reward_0(self, acf_coh, acf_fwhm, os_fwhm):
		reward = 0.0

		acf_fhwm_max = 160
		acf_coh_max = 1
		os_fwhm_max = 10.0
		acf_coh_penalty_start = 0.2
		sigmoid_penalty = 0.2
		sigmoid_scaler = 5.0

		reward += (sigmoid((-acf_fwhm + self.acf_fwhm) / acf_fhwm_max * sigmoid_scaler) - sigmoid_penalty)
		reward += (sigmoid((-acf_coh + self.acf_coh) / acf_coh_max * sigmoid_scaler) - sigmoid_penalty) * 2.0
		#reward += (sigmoid((os_fwhm - self.os_fwhm) / os_fwhm_max * sigmoid_scaler) - sigmoid_penalty)
		if acf_coh > 0.2:
			reward += - 1 / (1 - acf_coh_penalty_start) - acf_coh_penalty_start * 2.0
		reward += 1.0

		return reward

	def get_reward_1(self, acf_coh, acf_fwhm, os_fwhm, rf_med):
		reward = 0.0
		acf_coh_reward = 0.0
		acf_fwhm_reward = 0.0

		if acf_coh > 0.2:
			acf_coh_reward = - (acf_coh) * 2.0
		# # print('acf_coh_reward', acf_coh_reward)

		acf_fhwm_max = 160
		sigmoid_penalty = 0.3
		sigmoid_scaler = 5.0

		acf_fwhm_reward += (sigmoid((-acf_fwhm + self.acf_fwhm) / acf_fhwm_max * sigmoid_scaler) - sigmoid_penalty)
		# # print('acf_fwhm_reward', acf_fwhm_reward)

		reward = acf_coh_reward + acf_fwhm_reward

		return reward

	def get_reward_2(self, acf_fwhm):
		print(acf_fwhm, end='\r')
		reward = -1.0
		# reward += acf_fwhm / 5

		fwhm_good_enough_value = 9.89
		# fwhm_good_enough_value = 4.59
		if acf_fwhm > fwhm_good_enough_value:
			reward = 200.0

		return reward