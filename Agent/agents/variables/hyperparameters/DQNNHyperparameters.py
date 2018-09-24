import tensorflow as tf

from agents.variables.hyperparameters.NNHyperparameters import NNHyperparameters

class DQNNHyperparameters(NNHyperparameters):
	def __init__(self, *args, **kwargs):
		super(DQNNHyperparameters, self).__init__(*args, **kwargs)
		learning_rate = None
		self.batch_size = None #How many experiences to use for each training step.
		self.update_freq = None #How often to perform a training step.
		self.y = None #Discount factor on the target Q-values
		self.startE = None #Starting chance of random action
		self.endE = None #Final chance of random action
		self.annealing_steps = None #How many steps of training to reduce startE to endE.
		self.num_episodes = None #How many episodes of game environment to train network with.
		self.pre_train_steps = None #How many steps of random actions before training begins.
		self.max_epLength = None #The max allowed length of our episode.
		self.h_size = None #The size of the final convolutional layer before splitting it into Advantage and Value streams.
		self.tau = None #Rate to update target network toward primary network
		print('DQNNHyperparameters inited!')