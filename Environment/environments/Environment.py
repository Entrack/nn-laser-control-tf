
class Environment():
	def __init__(self):
		self.state = 0		
		self.num_states = self.get_num_states()
		self.num_actions = self.get_num_actions()

		self.selected_state = 0
		self.result = 0
		self.reward = 0
		print('Environment inited!')

	def get_state(self):
		return

	def act_and_get_reward(self, action):
		return

	def peform_action(self, action):
		return

	def get_result(self):
		return

	def get_reward(self):
		return

	def get_num_states(self):
		return

	def get_num_actions(self):
		return