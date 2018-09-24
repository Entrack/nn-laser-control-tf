from agents.variables.networks.NN import NN

import tensorflow as tf
import tensorflow.contrib.slim as slim

class DQNNGridWorld(NN):
	def __init__(self, hyperparameters, environment, *args, **kwargs):
		super(DQNNGridWorld, self).__init__(*args, **kwargs)

		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		self.scalarInput =  tf.placeholder(shape=[None,environment.state_size],dtype=tf.float32)
		imageIn = tf.reshape(self.scalarInput,shape=[-1,environment.one_dim_state_length,environment.one_dim_state_length,3])
		conv1 = slim.conv2d( \
		    inputs=imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
		conv2 = slim.conv2d( \
		    inputs=conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
		conv3 = slim.conv2d( \
		    inputs=conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
		conv4 = slim.conv2d( \
		    inputs=conv3,num_outputs=hyperparameters.h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)		

		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		streamAC, streamVC = tf.split(conv4,2,3)
		streamA = slim.flatten(streamAC)
		streamV = slim.flatten(streamVC)
		xavier_init = tf.contrib.layers.xavier_initializer()
		AW = tf.Variable(xavier_init([hyperparameters.h_size//2,environment.num_actions]))
		# AW = tf.Variable(xavier_init([hyperparameters.h_size//2,environment.actions]))
		VW = tf.Variable(xavier_init([hyperparameters.h_size//2,1]))
		Advantage = tf.matmul(streamA,AW)
		Value = tf.matmul(streamV,VW)

		#Then combine them together to get our final Q-values.
		self.Qout = Value + tf.subtract(Advantage,tf.reduce_mean(Advantage,axis=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		actions_onehot = tf.one_hot(self.actions,environment.num_actions,dtype=tf.float32)
		# actions_onehot = tf.one_hot(self.actions,environment.actions,dtype=tf.float32)

		Q = tf.reduce_sum(tf.multiply(self.Qout, actions_onehot), axis=1)

		td_error = tf.square(self.targetQ - Q)
		loss = tf.reduce_mean(td_error)
		trainer = tf.train.AdamOptimizer(learning_rate=hyperparameters.learning_rate)
		self.updateModel = trainer.minimize(loss)
		
		print('DQNNGridWorld', 'inited!')