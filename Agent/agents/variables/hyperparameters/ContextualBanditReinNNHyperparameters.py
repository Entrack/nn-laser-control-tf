import tensorflow as tf

from agents.variables.hyperparameters.NNHyperparameters import NNHyperparameters

class ContextualBanditReinNNHyperparameters(NNHyperparameters):
	def __init__(self, *args, **kwargs):
		super(ContextualBanditReinNNHyperparameters, self).__init__(*args, **kwargs)
		self.learning_rate = None
		self.eps = None
		self.num_episodes = None
		print('ContextualBanditReinNNHyperparameters inited!')