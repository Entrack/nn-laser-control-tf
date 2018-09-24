import tensorflow as tf

from agents.variables.structures.NNStructure import NNStructure

class ContextualBanditReinNNStructure(NNStructure):
	def __init__(self, *args, **kwargs):
		super(ContextualBanditReinNNStructure, self).__init__(*args, **kwargs)
		tf.reset_default_graph()
		self.state_in= None
		self.chosen_action = None
		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = None
		self.action_holder = None
		self.responsible_weight = None
		self.loss = None
		self.update = None
		print('ContextualBanditReinNNStructure inited!')