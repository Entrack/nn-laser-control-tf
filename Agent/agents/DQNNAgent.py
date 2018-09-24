from __future__ import division

#import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
import time



from agents.Agent import Agent
from agents.modules.ExperienceBuffer import ExperienceBuffer as experience_buffer

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class DQNNAgent(Agent):
	def __init__(self, hyperparameters, nn_architechture, environment, 
		if_load_model = False, if_save_model = False,
		model_path = './dqn', 
		is_infinite = False,
		if_train = True, 
		model_saving_period = 5, statistics_print_period = 10,
		reward_list_print_number = 10, rMat_resize_number = 100,
		*args, **kwargs):
		super(DQNNAgent, self).__init__(*args, **kwargs)
		self.hp = hyperparameters
		self.check_structure(self.hp)
		self.environment = environment
		self.if_load_model = if_load_model
		self.if_save_model = if_save_model
		self.is_infinite = is_infinite
		self.if_train = if_train
		self.model_path = model_path
		self.nn_architechture = nn_architechture

		self.mainQN = None
		self.targetQN = None
		self.init_nns()

		self.sess = None
		self.init = None
		self.saver = tf.train.Saver()
		self.trainables = tf.trainable_variables()
		self.targetOps = self.updateTargetGraph(self.trainables,self.hp.tau)
		self.myBuffer = experience_buffer()

		self.e = 0
		self.stepDrop = 0
		self.init_e()

		self.total_steps = None
		self.init_total_steps()

		self.jList = []
		self.rList = []

		self.current_episode = 0
		self.episode_buffer = None
		self.state = None
		self.new_state = None
		self.episode_done = False
		self.reward = 0
		self.rAll = 0
		self.j = 0

		self.model_saving_period = model_saving_period
		self.statistics_print_period = statistics_print_period
		self.reward_list_print_number = reward_list_print_number
		self.rMat_resize_number = rMat_resize_number

		self.create_save_folder()
		self.init_session()
		self.init_variables()
		print('DQNNAgent', 'inited!')

	def init_nns(self):
		tf.reset_default_graph()
		self.mainQN = self.nn_architechture(self.hp, self.environment)
		self.targetQN = self.nn_architechture(self.hp, self.environment)

	def init_e(self):
		self.e = self.hp.startE
		self.stepDrop = (self.hp.startE - self.hp.endE) / self.hp.annealing_steps

	def init_total_steps(self):
		# with tf.Session() as sess:
		self.total_steps = tf.Variable(0, name='total_steps', trainable=True, dtype=tf.int64)
		# self.total_steps = tf.get_variable("total_steps", shape=[1], initializer = tf.zeros_initializer)

	def get_total_steps(self):
		return self.sess.run(self.total_steps)

	def increment_total_steps(self):
		self.sess.run(tf.assign(self.total_steps, self.total_steps + 1))

	def create_save_folder(self):
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

	def init_session(self):
		self.sess = tf.Session()

	def init_variables(self):
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

	def run(self):
		self.load_model()
		while self.current_episode < self.hp.num_episodes or self.environment.is_infinite:
			self.run_q_nn()
			self.current_episode += 1
		self.save_model()
		self.show_statistics()

	def load_model(self):
		if self.if_load_model == True:
			print('Loading Model...')
			try:
				ckpt = tf.train.get_checkpoint_state(self.model_path)
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
			except:
				print('Failed to load the model')

	# def load_model(self):
	# 	if self.if_load_model == True:
	# 		print('Loading Model...')
	# 		ckpt = tf.train.get_checkpoint_state(self.model_path)
	# 		self.saver.restore(self.sess,ckpt.model_checkpoint_path)
	# 		# chkp.print_tensors_in_checkpoint_file(self.model_path + "/model.ckpt", tensor_name='total_steps', all_tensors=False)
	# 		print_tensors_in_checkpoint_file(file_name=self.model_path + "/model.ckpt", tensor_name='', all_tensors=False)
	# 		exit(0)

	def run_q_nn(self):
		print('')
		print('running episode #', self.current_episode)
		self.init_episode_buffer()
		self.get_state()
		self.reset_run_info()
		self.run_episode_loop()

	def init_episode_buffer(self):
		self.episode_buffer = experience_buffer()

	def get_state(self):
		state = self.environment.reset()
		self.state = self.flatten_state(state)
		return self.state

	def flatten_state(self, states):
		return np.reshape(states,[self.environment.state_size])

	def reset_run_info(self):
		self.episode_done = False
		self.rAll = 0
		self.j = 0

	def run_episode_loop(self):
		while self.j < self.hp.max_epLength or self.is_infinite:
			self.j+=1
			self.run_episode()
			# print('running episode #', self.current_episode)
			if self.episode_done == True:
				if self.is_infinite and self.get_total_steps() > self.hp.pre_train_steps:
					print('System convirged')
					exit(0)
				break
		self.save_run_info()
		self.save_model_if_needed()
		self.print_statistics_if_needed()
		
	def run_episode(self):
		# print('self.get_total_steps()', self.get_total_steps())
		print('Step:\t', self.get_total_steps(), end='\r')
		self.choose_action()
		self.step()
		self.train_if_needed()			
		self.update_run_info()

	def choose_action(self):
		#Choose an action by greedily (with e chance of random action) from the Q-network
		if np.random.rand(1) < self.e or self.get_total_steps() < self.hp.pre_train_steps:
			self.action = np.random.randint(0,self.environment.num_actions)
			# print('Choosing by random')
		else:
			self.action = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:[self.state]})[0]
			# print('action number:', self.action)
			# print('Choosing by neural network')

	def step(self):
		self.new_state,self.reward,self.episode_done = self.environment.step(self.action)
		self.new_state = self.flatten_state(self.new_state)
		self.increment_total_steps()
		self.save_experience()

	def save_experience(self):
		self.episode_buffer.add(np.reshape(np.array([self.state,self.action,
		self.reward,self.new_state,self.episode_done]),[1,5])) #Save the experience to our episode buffer.		

	def train_if_needed(self):
		if self.get_total_steps() > self.hp.pre_train_steps:
			self.train()

	def train(self):
		self.decrease_e_if_needed()
		self.update_nn_if_needed()

	def decrease_e_if_needed(self):
		if self.e > self.hp.endE:
			self.decrease_e()

	def decrease_e(self):
		self.e -= self.stepDrop	

	def update_nn_if_needed(self):
		if self.get_total_steps() % (self.hp.update_freq) == 0 and self.if_train:
			self.update_nn()	

	def update_nn(self):
		#Get a random batch of experiences.
		trainBatch = self.myBuffer.sample(self.hp.batch_size) 
		new_state_vector = trainBatch[:,3]
		#Below we perform the Double-DQN update to the target Q-values
		Q1 = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:np.vstack(new_state_vector)})
		Q2 = self.sess.run(self.targetQN.Qout,feed_dict={self.targetQN.scalarInput:np.vstack(new_state_vector)})
		done_vector = trainBatch[:,4]
		end_multiplier = -(done_vector - 1)
		doubleQ = Q2[range(self.hp.batch_size),Q1]
		reward_vector = trainBatch[:,2]
		targetQ = reward_vector + (self.hp.y * doubleQ * end_multiplier)
		#Update the network with our target values.
		state_vector = trainBatch[:,0]
		action_vector = trainBatch[:,1]
		_ = self.sess.run(self.mainQN.updateModel, \
			feed_dict={self.mainQN.scalarInput:np.vstack(state_vector),
			self.mainQN.targetQ:targetQ, self.mainQN.actions:action_vector})
		
		#Update the target network toward the primary network.
		self.updateTarget(self.targetOps,self.sess) 

	def update_run_info(self):
		self.rAll += self.reward
		self.state = self.new_state

	def save_run_info(self):
		self.myBuffer.add(self.episode_buffer.buffer)
		self.jList.append(self.j)
		self.rList.append(self.rAll)

	def save_model_if_needed(self):
		if self.current_episode % self.model_saving_period == 0 and not self.current_episode is 0:
			self.save_model()

	def save_model(self):
		if self.if_save_model:
			file_name = '/model-' + str(self.current_episode) + '.ckpt'
			# file_name = '/model' + '.ckpt'
			self.saver.save(self.sess,self.model_path + file_name)
			print('Model saved as:', file_name)

	def print_statistics_if_needed(self):
		if len(self.rList) % self.statistics_print_period == 0:
			self.print_statistics()

	def print_statistics(self):
		print(self.get_total_steps(), np.mean(self.rList[-self.reward_list_print_number:]), self.e)

	def updateTargetGraph(self, tfVars,tau):
		total_vars = len(tfVars)
		op_holder = []
		for idx,var in enumerate(tfVars[0:total_vars//2]):
			op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + 
				((1-tau)*tfVars[idx+total_vars//2].value())))
		return op_holder

	def updateTarget(self, op_holder,sess):
		for op in op_holder:
			sess.run(op)

	def show_statistics(self):
		print("Percent of succesful episodes: " + str(sum(self.rList)/self.hp.num_episodes) + "%")

		rMat = np.resize(np.array(self.rList), 
			[len(self.rList) // self.rMat_resize_number, self.rMat_resize_number])
		rMean = np.average(rMat,1)
		plt.plot(rMean)