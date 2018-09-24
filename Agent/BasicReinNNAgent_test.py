import gym
import numpy as np
import random
import tensorflow as tf

from agents.BasicReinNNAgent import BasicReinNNAgent as agent
from agents.variables.structures.BasicReinNNStructure import BasicReinNNStructure as nn_structure
from agents.variables.hyperparameters.BasicReinNNHyperparameters import BasicReinNNHyperparameters as nn_hyperparameters



example_nn_hp = nn_hyperparameters()
example_nn_hp.learning_rate = 0.2
example_nn_hp.y = 0.99
example_nn_hp.eps = 0.1
example_nn_hp.num_episodes = 500

example_nn = nn_structure()
example_nn.inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
example_nn.W = tf.Variable(tf.random_uniform([16,4],0,0.01))
example_nn.Qout = tf.matmul(example_nn.inputs1,example_nn.W)
example_nn.predict = tf.argmax(example_nn.Qout,1)
example_nn.nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
example_nn.loss = tf.reduce_sum(tf.square(example_nn.nextQ - example_nn.Qout))
example_nn.trainer = tf.train.GradientDescentOptimizer(learning_rate=example_nn_hp.learning_rate)
example_nn.updateModel = example_nn.trainer.minimize(example_nn.loss)



nn_architechture = example_nn
nn_hp = example_nn_hp

def test():
	print('testing')

	env = gym.make('FrozenLake-v0')

	rein_nn = agent(nn_architechture, nn_hp, env)

	rein_nn.run()


test()