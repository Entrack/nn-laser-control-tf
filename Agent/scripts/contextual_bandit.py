import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np



class contextual_bandit():
	def __init__(self):
		self.state = 0
		#List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
		#self.bandits = np.array([[0.2,0,-0.0,-5],[5,5,-5,5],[0.1,-5,1,0.25]])
		self.bandits = np.array([[0.1,-5,1,0.25],[0.1,-5,1,0.25],[0.1,-5,1,0.25]])
		self.num_bandits = self.bandits.shape[0]
		self.num_actions = self.bandits.shape[1]
		
	def getBandit(self):
		self.state = np.random.randint(0,len(self.bandits)) #Returns a random state for each episode.
		return self.state
		
	def pullArm(self,action):
		#Get a random number.
		bandit = self.bandits[self.state,action]
		result = np.random.randn(1)
		if result > bandit:
			#return a positive reward.
			return 1
		else:
			#return a negative reward.
			return -1



class agent():
	def __init__(self, lr, s_size,a_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)
		state_in_OH = slim.one_hot_encoding(self.state_in,s_size)

		# Сейчас задан один полносвязный слой
		# output = slim.fully_connected(state_in_OH,a_size,\
		# 	biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())

		# output = slim.fully_connected(state_in_OH,a_size,\
		# 	biases_initializer=None,activation_fn=None,weights_initializer=tf.ones_initializer())


		# Я хочу добавить второй полносвязный слой с, допустим, 32-мя вершинами
		# Правильно ли я это делаю и почему эффективность падает?
		output = slim.fully_connected(state_in_OH,3,\
			biases_initializer=None,activation_fn=None,weights_initializer=tf.ones_initializer())
		output = slim.fully_connected(output,a_size,\
			biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones_initializer())

		output = tf.reshape(output,[-1])
		self.chosen_action = tf.argmax(output,0)

		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
		self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
		self.responsible_weight = tf.slice(output,self.action_holder,[1])
		#self.responsible_weight = tf.slice(output,tf.cast(tf.reshape(self.chosen_action, [1]), dtype=tf.int32),[1])
		#self.responsible_weight = output[self.chosen_action]
		self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
		self.update = optimizer.minimize(self.loss)



tf.reset_default_graph() #Clear the Tensorflow graph.

cBandit = contextual_bandit() #Load the bandits.
myAgent = agent(lr=0.001,s_size=cBandit.num_bandits,a_size=cBandit.num_actions) #Load the agent.
#weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

# weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.
# print(weights.shape)	
# print(weights)	
# print(tf.trainable_variables())
# exit(0)

weights = tf.trainable_variables() #The weights we will evaluate to look into the network.
weights = tf.concat([weights[0], weights[1]], axis=0)

total_episodes = 10001 #Set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

init = tf.initialize_all_variables()

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:
		print('runnning episode number', i)
		s = cBandit.getBandit() #Get a state from the environment.
		
		#Choose either a random action or one from our network.
		if np.random.rand(1) < e:
			action = np.random.randint(cBandit.num_actions)
		else:
			action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]})
		
		reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.
		
		#Update the network.
		feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
		_,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
		
		#Update our running tally of scores.
		total_reward[s,action] += reward
		if i % 500 == 0:
			print("Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1)))
		i+=1
for a in range(cBandit.num_bandits):
	print(ww.shape)
	print(ww	)
	print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising....")
	if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
		print ("...and it was right!")
	else:
		print ("...and it was wrong!")