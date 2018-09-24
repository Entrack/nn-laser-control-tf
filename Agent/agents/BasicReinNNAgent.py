import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.Agent import Agent

class BasicReinNNAgent(Agent):
	def __init__(self, neural_network, hyperparameters, environment, *args, **kwargs):
		super(BasicReinNNAgent, self).__init__(*args, **kwargs)
		self.nn = neural_network
		self.check_structure(self.nn)
		self.hp = hyperparameters
		self.check_structure(self.hp)
		self.environment = environment
		self.last_episode = 0
		self.init = tf.global_variables_initializer()
		self.sess = None

		self.jList = []
		self.rList = []

		print('BasicReinNNAgent inited!')

	def run_session(self):
		self.sess = tf.Session()
		self.sess.run(self.init)

	def run(self):
		self.run_session()

		for i in range(self.last_episode, self.hp.num_episodes):
			print('Runnning episode number', i)
			#The Q-Network
			j, rAll = self.run_Q_nn(i, 99)
			print('J:', j)
			self.jList.append(j)
			self.rList.append(rAll)
		print("Percent of succesful episodes: " + str(sum(self.rList)/self.hp.num_episodes) + "%")

	def run_Q_nn(self, i, j_max):
		#Reset environment and get first new observation
		s = self.environment.reset()
		rAll = 0
		d = False
		j = 0
		while j < j_max:
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			a,allQ = self.sess.run([self.nn.predict,self.nn.Qout],feed_dict={self.nn.inputs1:np.identity(16)[s:s+1]})
			if np.random.rand(1) < self.hp.eps:
				a[0] = self.environment.action_space.sample()
			#Get new state and reward from environment
			s1,r,d,_ = self.environment.step(a[0])
			#Obtain the Q' values by feeding the new state through our network
			Q1 = self.sess.run(self.nn.Qout,feed_dict={self.nn.inputs1:np.identity(16)[s1:s1+1]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0,a[0]] = r + self.hp.y*maxQ1
			#Train our network using target and predicted Q values
			_,W1 = self.sess.run([self.nn.updateModel,self.nn.W],feed_dict={self.nn.inputs1:np.identity(16)[s:s+1],self.nn.nextQ:targetQ})
			rAll += r
			s = s1
			if d == True:
				#Reduce chance of random action as we train the model.
				self.hp.eps = 1./((i/50) + 10)
				break
		return j, rAll