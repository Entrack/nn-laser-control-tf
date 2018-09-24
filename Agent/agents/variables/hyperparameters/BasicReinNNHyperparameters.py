import tensorflow as tf

from agents.variables.hyperparameters.NNHyperparameters import NNHyperparameters

class BasicReinNNHyperparameters(NNHyperparameters):
	def __init__(self, *args, **kwargs):
		super(BasicReinNNHyperparameters, self).__init__(*args, **kwargs)
		self.learning_rate = None
		self.y = None
		self.eps = None
		self.num_episodes = None
		print('BasicReinNNHyperparameters inited!')