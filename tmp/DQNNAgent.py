from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os



from agents.Agent import Agent
from agents.variables.networks.DQNN import DQNN
from agents.modules.ExperienceBuffer import ExperienceBuffer as experience_buffer

class DQNNAgent(Agent):
	def __init__(self, hyperparameters, environment, if_load_model, model_path, *args, **kwargs):
		super(DQNNAgent, self).__init__(*args, **kwargs)
		self.hp = hyperparameters
		self.check_structure(self.hp)
		self.environment = environment
		self.if_load_model = if_load_model
		self.model_path = model_path

		self.mainQN = None
		self.targetQN = None
		self.init_nns()

		self.sess = None
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.trainables = tf.trainable_variables()
		self.targetOps = self.updateTargetGraph(self.trainables,self.hp.tau)
		self.myBuffer = experience_buffer()

		self.e = self.hp.startE
		self.stepDrop = (self.hp.startE - self.hp.endE)/self.hp.annealing_steps

		self.jList = []
		self.rList = []
		self.total_steps = 0
		self.current_episode = 0

		self.create_save_folder()
		self.run_session()
		print('DQNNAgent', 'inited!')

	def init_nns(self):
		tf.reset_default_graph()
		self.mainQN = DQNN(self.hp, self.environment)
		self.targetQN = DQNN(self.hp, self.environment)

	def create_save_folder(self):
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

	def run_session(self):
		self.sess = tf.Session()
		self.sess.run(self.init)

	def run(self):
		self.run_session()

		self.load_model()
		while self.current_episode < self.hp.num_episodes:
			self.run_q_nn()
			self.current_episode += 1

		self.save_model()
		self.show_statistics()

	def load_model(self):
		if self.if_load_model == True:
			print('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(self.model_path)
			self.saver.restore(self.sess,ckpt.model_checkpoint_path)

	def run_q_nn(self):
		print('running episode #', self.current_episode)
		episodeBuffer = experience_buffer()
		#Reset environment and get first new observation
		s = self.environment.reset()
		s = self.processState(s)
		d = False
		rAll = 0
		j = 0
		#The Q-Network
		while j < self.hp.max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			if np.random.rand(1) < self.e or self.total_steps < self.hp.pre_train_steps:
				a = np.random.randint(0,4)
			else:
				a = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:[s]})[0]
			s1,r,d = self.environment.step(a)
			s1 = self.processState(s1)
			self.total_steps += 1
			episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
			
			if self.total_steps > self.hp.pre_train_steps:
				if self.e > self.hp.endE:
					self.e -= self.stepDrop
				
				if self.total_steps % (self.hp.update_freq) == 0:
					trainBatch = self.myBuffer.sample(self.hp.batch_size) #Get a random batch of experiences.
					#Below we perform the Double-DQN update to the target Q-values
					Q1 = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,3])})
					Q2 = self.sess.run(self.targetQN.Qout,feed_dict={self.targetQN.scalarInput:np.vstack(trainBatch[:,3])})
					end_multiplier = -(trainBatch[:,4] - 1)
					doubleQ = Q2[range(self.hp.batch_size),Q1]
					targetQ = trainBatch[:,2] + (self.hp.y*doubleQ * end_multiplier)
					#Update the network with our target values.
					_ = self.sess.run(self.mainQN.updateModel, \
						feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,0]),self.mainQN.targetQ:targetQ, self.mainQN.actions:trainBatch[:,1]})
					
					self.updateTarget(self.targetOps,self.sess) #Update the target network toward the primary network.
			rAll += r
			s = s1
			
			if d == True:

				break
		
		self.myBuffer.add(episodeBuffer.buffer)
		self.jList.append(j)
		self.rList.append(rAll)
		#Periodically save the model. 
		if self.current_episode % 1000 == 0:
			self.saver.save(self.sess,self.model_path+'/model-'+str(self.current_episode)+'.ckpt')
			print("Saved Model")
		if len(self.rList) % 10 == 0:
			print(self.total_steps,np.mean(self.rList[-10:]), self.e)


	def processState(self, states):
		return np.reshape(states,[21168])

	def updateTargetGraph(self, tfVars,tau):
		total_vars = len(tfVars)
		op_holder = []
		for idx,var in enumerate(tfVars[0:total_vars//2]):
			op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
		return op_holder

	def updateTarget(self, op_holder,sess):
		for op in op_holder:
			sess.run(op)

	def save_model(self):
		self.saver.save(self.sess,self.model_path+'/model-'+str(i)+'.ckpt')

	def show_statistics(self):
		print("Percent of succesful episodes: " + str(sum(self.rList)/self.hp.num_episodes) + "%")

		rMat = np.resize(np.array(self.rList),[len(self.rList)//100,100])
		rMean = np.average(rMat,1)
		plt.plot(rMean)