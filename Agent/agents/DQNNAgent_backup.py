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

def processState(states):
	return np.reshape(states,[21168])



def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

class DQNNAgent(Agent):
	def __init__(self, hyperparameters, environment, if_load_model, model_path, *args, **kwargs):
		super(DQNNAgent, self).__init__(*args, **kwargs)
		self.hp = hyperparameters
		self.check_structure(self.hp)
		self.environment = environment
		self.if_load_model = if_load_model
		self.model_path = model_path

	def run(self):
		tf.reset_default_graph()
		mainQN = DQNN(self.hp, self.environment)
		targetQN = DQNN(self.hp, self.environment)

		init = tf.global_variables_initializer()

		saver = tf.train.Saver()

		trainables = tf.trainable_variables()

		targetOps = updateTargetGraph(trainables,self.hp.tau)

		myBuffer = experience_buffer()

		#Set the rate of random action decrease. 
		e = self.hp.startE
		stepDrop = (self.hp.startE - self.hp.endE)/self.hp.annealing_steps

		#create lists to contain total rewards and steps per episode
		jList = []
		rList = []
		total_steps = 0

		#Make a path for our model to be saved in.
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

		with tf.Session() as sess:
			sess.run(init)
			#
			writer = tf.summary.FileWriter('logs', sess.graph)
			#
			if self.if_load_model == True:
				print('Loading Model...')
				ckpt = tf.train.get_checkpoint_state(self.model_path)
				saver.restore(sess,ckpt.model_checkpoint_path)
			for i in range(self.hp.num_episodes):
				print('running episode #', i)
				episodeBuffer = experience_buffer()
				#Reset environment and get first new observation
				s = self.environment.reset()
				s = processState(s)
				d = False
				rAll = 0
				j = 0
				#The Q-Network
				while j < self.hp.max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
					j+=1
					#Choose an action by greedily (with e chance of random action) from the Q-network
					if np.random.rand(1) < e or total_steps < self.hp.pre_train_steps:
						a = np.random.randint(0,4)
					else:
						a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
					s1,r,d = self.environment.step(a)
					s1 = processState(s1)
					total_steps += 1
					episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
					
					if total_steps > self.hp.pre_train_steps:
						if e > self.hp.endE:
							e -= stepDrop
						
						if total_steps % (self.hp.update_freq) == 0:
							trainBatch = myBuffer.sample(self.hp.batch_size) #Get a random batch of experiences.
							#Below we perform the Double-DQN update to the target Q-values
							Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
							Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
							end_multiplier = -(trainBatch[:,4] - 1)
							doubleQ = Q2[range(self.hp.batch_size),Q1]
							targetQ = trainBatch[:,2] + (self.hp.y*doubleQ * end_multiplier)
							#Update the network with our target values.
							_ = sess.run(mainQN.updateModel, \
								feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
							
							updateTarget(targetOps,sess) #Update the target network toward the primary network.
					rAll += r
					s = s1
					
					if d == True:

						break
				
				myBuffer.add(episodeBuffer.buffer)
				jList.append(j)
				rList.append(rAll)
				#Periodically save the model. 
				if i % 1000 == 0:
					saver.save(sess,self.model_path+'/model-'+str(i)+'.ckpt')
					print("Saved Model")
				if len(rList) % 10 == 0:
					print(total_steps,np.mean(rList[-10:]), e)
			saver.save(sess,self.model_path+'/model-'+str(i)+'.ckpt')
		print("Percent of succesful episodes: " + str(sum(rList)/self.hp.num_episodes) + "%")



		rMat = np.resize(np.array(rList),[len(rList)//100,100])
		rMean = np.average(rMat,1)
		plt.plot(rMean)