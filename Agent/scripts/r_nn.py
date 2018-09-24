import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt



env = gym.make('FrozenLake-v0')
tf.reset_default_graph()



class NNStructure():
    def __init__(self):
        tf.reset_default_graph()
        self.inputs1 = None
        self.W = None
        self.Qout = None
        self.predict = None
        self.nextQ = None
        self.loss = None
        self.trainer = None
        self.updateModel = None
        print('NNStructure inited!')

    def is_filled(self):
        isFilled = True
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for var in members:
            print(var)
            if getattr(self, var) is None:
                isFilled = False
        return isFilled


example_nn = NNStructure()

example_nn.inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
example_nn.W = tf.Variable(tf.random_uniform([16,4],0,0.01))
example_nn.Qout = tf.matmul(example_nn.inputs1,example_nn.W)
example_nn.predict = tf.argmax(example_nn.Qout,1)

example_nn.nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
example_nn.loss = tf.reduce_sum(tf.square(example_nn.nextQ - example_nn.Qout))
example_nn.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
example_nn.updateModel = example_nn.trainer.minimize(example_nn.loss)
#example_nn.updateModel = None

nn = example_nn

inputs1 = nn.inputs1
W = nn.W
Qout = nn.Qout
predict = nn.predict

nextQ = nn.nextQ
loss = nn.loss
trainer = nn.trainer
updateModel = nn.updateModel

print('Inited NN')

if nn.is_filled():
    print('Filled')
else:
    print('Not filled')
exit(0)

# #These lines establish the feed-forward part of the network used to choose actions
# inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
# Qout = tf.matmul(inputs1,W)
# predict = tf.argmax(Qout,1)

# #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
# nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ - Qout))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# updateModel = trainer.minimize(loss)



init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        print('episode number', i)
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")



print(rList[0::100])



print(jList[0::100])