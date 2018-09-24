'''
To use this script,
fill config.txt with the following values (without numbers in brackets):
(0) Environment Lib path

'''

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

from environments.MultiSlotMachineEnvironment import MultiSlotMachineEnvironment as test_environment
from agents.ContextualBanditReinNNAgent import ContextualBanditReinNNAgent as agent
from agents.variables.structures.ContextualBanditReinNNStructure import ContextualBanditReinNNStructure as nn_structure
from agents.variables.hyperparameters.ContextualBanditReinNNHyperparameters import ContextualBanditReinNNHyperparameters as nn_hyperparameters

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np



environment = test_environment()

example_nn_hp = nn_hyperparameters()
example_nn_hp.learning_rate = 0.2
example_nn_hp.eps = 0.1
example_nn_hp.num_episodes = 10000

example_nn = nn_structure()
example_nn.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
state_in_OH = slim.one_hot_encoding(example_nn.state_in,environment.num_states)
output = slim.fully_connected(state_in_OH,environment.num_actions,\
	biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())
example_nn.output = tf.reshape(output,[-1])
example_nn.chosen_action = tf.argmax(example_nn.output,0)
#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.
example_nn.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
example_nn.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
example_nn.responsible_weight = tf.slice(example_nn.output,example_nn.action_holder,[1])
example_nn.loss = -(tf.log(example_nn.responsible_weight)*example_nn.reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=example_nn_hp.learning_rate)
example_nn.update = optimizer.minimize(example_nn.loss)



nn_architechture = example_nn
nn_hp = example_nn_hp

def test():
	rein_nn = agent(nn_architechture, nn_hp, environment)

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