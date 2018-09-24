import numpy as np

from environments.Environment import Environment

class MultiSlotMachineEnvironment(Environment):
	def __init__(self):
		#List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
		self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])

		self.state = 0		
		self.num_states = self.get_num_states()
		self.num_actions = self.get_num_actions()

		self.selected_state = 0
		self.result = 0
		self.reward = 0
		print('MultiSlotMachineEnvironment inited!')

	def get_state(self):
		self.state = np.random.randint(0,len(self.bandits)) #Returns a random state for each episode.
		return self.state

	def act_and_get_reward(self, action):
		self.peform_action(action)
		self.get_result()
		return self.get_reward()

	def peform_action(self, action):
		#Get a random number.
		self.selected_state = self.bandits[self.state,action]

	def get_result(self):
		self.result = np.random.randn(1)

	def get_reward(self):
		if self.result > self.selected_state:
			self.result = 1
		else:
			self.result = -1
		return self.result

	def get_num_states(self):
		return self.bandits.shape[0]

	def get_num_actions(self):
		return self.bandits.shape[1]