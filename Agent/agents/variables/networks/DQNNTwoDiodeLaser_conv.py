from agents.variables.networks.NN import NN

import tensorflow as tf
import tensorflow.contrib.slim as slim

class DQNNTwoDiodeLaser(NN):
	def __init__(self, hyperparameters, environment, *args, **kwargs):
		super(DQNNTwoDiodeLaser, self).__init__(*args, **kwargs)

		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		self.scalarInput =  tf.placeholder(shape=[None,environment.state_size],dtype=tf.float32)
		fully_conn1 = slim.fully_connected(inputs=self.scalarInput,num_outputs=16,\
		  biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())
		print(fully_conn1.get_shape())
		imageIn = tf.reshape(fully_conn1,shape=[-1,4,4,1])
		print(imageIn.get_shape())
		conv1 = slim.conv2d( \
		    # inputs=imageIn,num_outputs=hyperparameters.h_size,kernel_size=[2,2],stride=[1,1],padding='VALID', biases_initializer=None)
			inputs=imageIn,num_outputs=hyperparameters.h_size,kernel_size=[2,2],stride=[1,1],padding='SAME', biases_initializer=None)

		print(conv1.get_shape())
		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		streamAC, streamVC = tf.split(conv1,2,3)
		print(streamAC.get_shape())
		print(streamVC.get_shape())
		streamA = slim.flatten(streamAC)
		streamV = slim.flatten(streamVC)
		print(streamA.get_shape())
		print(streamV.get_shape())
		xavier_init = tf.contrib.layers.xavier_initializer()
		# AW = tf.Variable(xavier_init([hyperparameters.h_size//2,environment.num_actions]))
		# VW = tf.Variable(xavier_init([hyperparameters.h_size//2,1]))
		AW = tf.Variable(xavier_init([32//2,environment.num_actions]))
		VW = tf.Variable(xavier_init([32//2,1]))
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
		
		print(self.__class__.__name__, 'inited!')