import sys

'''
To use this script,
fill config.txt with the following values (without numbers in brackets):
(0) Measurement Lib path
(1) Serial Laser Control Lib path
(2) Signal Processing Lib path
(3) File Manager Lib path
(4) First laser diode serial port
(5) Second laser diode serial port
(6) Path to the folder for saving experimental data
(7) Reinforcement Learning Agnets Lib path
(8) Reinforcement Learning Environments Lib path
(9) Pre-made measurements folder path // required if you run FileVirtualDevice

'''

# LOADING CONFIG
def load_config():
	with open('config.cfg') as config:
		return [line.strip() for line in config.readlines()]

try:
	config = load_config()
	experimental_data_saving_folder = config[6]
	reinforcement_learning_agents_lib_path = config[7]
	reinforcement_learning_environments_lib_path = config[8]
except:
	raise Exception("Unable to read data from config.cfg" + " \n" +
		"See the script's head to see how fill in config file")

# Path to Reinforcement Learning Agents Lib
sys.path.insert(0, reinforcement_learning_agents_lib_path)
# Path to Reinforcement Learning Environments Lib
sys.path.insert(0, reinforcement_learning_environments_lib_path)

from Agent.agents.DQNNAgent import DQNNAgent as selected_agent
from Environment.environments.TwoDiodeLaserEnvironment import TwoDiodeLaserEnvironment as selected_environment
# from Agent.agents.variables.networks.DQNNTwoDiodeLaser import DQNNTwoDiodeLaser as nn_architechture
from Agent.agents.variables.networks.DQNNTwoDiodeLaser_simple import DQNNTwoDiodeLaser as nn_architechture
# from Agent.agents.variables.networks.DQNNTwoDiodeLaser_conv import DQNNTwoDiodeLaser as nn_architechture
# from Agent.agents.variables.networks.DQNNGridWorld import DQNNGridWorld as nn_architechture
from Agent.agents.variables.hyperparameters.DQNNHyperparameters import DQNNHyperparameters as selected_hyperparameters


if_learn = False 

hyperparameters = selected_hyperparameters()
hyperparameters.learning_rate = 0.0001#0.01#0.001#0.001 #0.0001
hyperparameters.batch_size = 32 #How many experiences to use for each training step.
hyperparameters.update_freq = 100#8#16 #How often to perform a training step.
hyperparameters.y = 0.99#0.5#.99 #Discount factor on the target Q-values
if if_learn:
	hyperparameters.startE = 0.3#Starting chance of random action
	hyperparameters.endE = 0.1 #Final chance of random action
	hyperparameters.annealing_steps = 3000#30000#2000#10000. #How many steps of training to reduce startE to endE.
	hyperparameters.pre_train_steps = 1000#600#120#200#50 #How many steps of random actions before training begins.
else:
	hyperparameters.startE = 0#Starting chance of random action
	hyperparameters.endE = 0#Final chance of random action
	hyperparameters.annealing_steps = 1#10000. #How many steps of training to reduce startE to endE.
	hyperparameters.pre_train_steps = 20#200#50 #How many steps of random actions before training begins.
hyperparameters.num_episodes = 2000#160 #How many episodes of game environment to train network with.
hyperparameters.max_epLength = 150 #The max allowed length of our episode.
hyperparameters.h_size = 8#6#16 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
hyperparameters.tau =  0.001#0.1#0.001#0.01 #Rate to update target network toward primary network


if if_learn:
	environment = selected_environment('./measurements/', if_save_pics = False, if_save_param_logs = False, if_penalize = True)
	agent = selected_agent(hyperparameters = hyperparameters, nn_architechture = nn_architechture, 
			environment = environment, 
			if_load_model = False, if_save_model = True,
			model_path = './dqn', is_infinite = False, if_train = True, 
			model_saving_period = 50, statistics_print_period = 1,
			reward_list_print_number = 10, rMat_resize_number = 100)
else:
	environment = selected_environment('./measurements/', if_save_pics = True, if_save_param_logs = True, if_penalize = True)
	agent = selected_agent(hyperparameters = hyperparameters, nn_architechture = nn_architechture, 
			environment = environment, 
			if_load_model = True, if_save_model = False,
			model_path = './dqn', is_infinite = True, if_train = False,
			model_saving_period = 50, statistics_print_period = 1,
			reward_list_print_number = 10, rMat_resize_number = 100)

def find_optimal_regime():
	agent.run()

find_optimal_regime()