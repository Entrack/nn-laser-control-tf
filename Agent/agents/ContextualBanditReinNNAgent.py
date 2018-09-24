import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.Agent import Agent

class ContextualBanditReinNNAgent(Agent):
	def __init__(self, neural_network, hyperparameters, environment, *args, **kwargs):
		super(ContextualBanditReinNNAgent, self).__init__(*args, **kwargs)
		self.nn = neural_network
		self.check_structure(self.nn)
		self.hp = hyperparameters
		self.check_structure(self.hp)
		self.environment = environment

		self.weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
		self.init = tf.global_variables_initializer()
		self.sess = None

		self.total_reward = np.zeros([self.environment.num_states,self.environment.num_actions]) #Set scoreboard for bandits to 0.

		self.episode_number = 0
		self.actions_weights = None
		self.state = None
		self.action = None
		self.reward = None

		self.j = 0

		self.run_session()

	def run(self):
		self.run_session()

		while self.episode_number < self.hp.num_episodes:
			self.run_q_nn()
			self.episode_number+=1
		self.predict()

	def run_session(self):
		self.sess = tf.Session()
		self.sess.run(self.init)

	def run_q_nn(self):
		print('runnning episode number', self.episode_number)

		self.get_state()
		
		self.choose_action()
		
		reward = self.get_reward()
		if reward > 0:
			self.j += 1
		print(chr(27) + "[2J")
		print(self.j / (self.episode_number + 1))
		
		self.update_network()

	def get_state(self):
		self.state = self.environment.get_state()
		return self.state

	def choose_action(self):
		#Choose either a random action or one from our network.
		if np.random.rand(1) < self.hp.eps:
			action = np.random.randint(self.environment.num_actions)
		else:
			action = self.sess.run(self.nn.chosen_action,feed_dict={self.nn.state_in:[self.state]})

		self.action = action
		return self.action

	def get_reward(self):
		self.reward = self.environment.act_and_get_reward(self.action) #Get our reward for taking an action given a bandit.
		return self.reward

	def update_network(self):
		#Update the network.
		feed_dict={self.nn.reward_holder:[self.reward],self.nn.action_holder:[self.action],
		self.nn.state_in:[self.state]}
		_, self.actions_weights = self.sess.run([self.nn.update,self.weights], feed_dict=feed_dict)
		return _, self.actions_weights

	def update_r_t_scores(self):
		#Update our running tally of scores.
		self.total_reward[self.state,self.action] += self.reward
		if episode_number % 500 == 0:
			print("Mean reward for each of the " + str(self.environment.num_states) + " bandits: " + str(np.mean(self.total_reward,axis=1)))

	def predict(self):
		for a in range(self.environment.num_states):
			print("The agent thinks action " + str(np.argmax(self.actions_weights[a])+1) + " for bandit " + str(a+1) + " is the most promising....")
			if np.argmax(self.actions_weights[a]) == np.argmin(self.environment.bandits[a]):
				print ("...and it was right!")
			else:
				print ("...and it was wrong!")