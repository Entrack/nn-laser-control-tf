# '''
# To use this script,
# fill config.txt with the following values (without numbers in brackets):
# (0) Environment Lib path

# '''

# LOADING CONFIG
def load_config():
	with open('config.cfg') as config:
		return [line.strip() for line in config.readlines()]

try:
	config = load_config()
	environment_lib_path = config[0]
except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")


import sys
# Environment Lib Path
sys.path.insert(0, environment_lib_path)

from environments.GridworldEnvironment import GridworldEnvironment as test_environment

from agents.DQNNAgent import DQNNAgent as agent
from agents.variables.hyperparameters.DQNNHyperparameters import DQNNHyperparameters as nn_hyperparameters
from agents.variables.networks.DQNNGridWorld import DQNNGridWorld as nn_architechture

import tensorflow as tf
import tensorflow.contrib.slim as slim



environment = test_environment(partial=False, size=3)



example_nn_hp = nn_hyperparameters()
example_nn_hp.batch_size = 32 #How many experiences to use for each training step.
example_nn_hp.update_freq = 4 #How often to perform a training step.
example_nn_hp.y = .99 #Discount factor on the target Q-values
example_nn_hp.startE = 1 #Starting chance of random action
example_nn_hp.endE = 0.1 #Final chance of random action
example_nn_hp.annealing_steps = 5000#10000. #How many steps of training to reduce startE to endE.
example_nn_hp.num_episodes = 2000#100 #How many episodes of game environment to train network with.
example_nn_hp.pre_train_steps = 2000#100 #How many steps of random actions before training begins.
example_nn_hp.max_epLength = 50 #The max allowed length of our episode.
example_nn_hp.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
example_nn_hp.tau = 0.001 #Rate to update target network toward primary network

example_nn_hp.learning_rate = 0.0001


nn_hp = example_nn_hp

def test():
	rein_nn = agent(nn_hp, nn_architechture, environment)

	rein_nn.run()

test()





























# example_nn = nn_structure()
# example_nn.state_in= tf.placeholder(shape=[1],dtype=tf.int32)

# state_in_OH = slim.one_hot_encoding(example_nn.state_in,s_size)
# w = slim.fully_connected(state_in_OH,16,\
# 	biases_initializer=None,activation_fn=None,weights_initializer=tf.ones_initializer(), scope="l_1")
# w = slim.fully_connected(w,a_size,\
# 	biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer(), scope="l_2")

# example_nn.output = tf.reshape(w,[-1])
# example_nn.chosen_action = tf.argmax(example_nn.output,0)

# #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
# #to compute the loss, and use it to update the network.
# example_nn.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
# example_nn.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
# example_nn.responsible_weight = tf.slice(example_nn.output,example_nn.action_holder,[1])
# example_nn.loss = -(tf.log(example_nn.responsible_weight)*example_nn.reward_holder)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
# example_nn.update = optimizer.minimize(example_nn.loss)