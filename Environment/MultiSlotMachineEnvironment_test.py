from environments.MultiSlotMachineEnvironment import MultiSlotMachineEnvironment

def test():
	print('testing')
	enviroment = MultiSlotMachineEnvironment()
	print('enviroment.get_state():', enviroment.get_state())
	action = 0
	print('enviroment.get_reward(action): ', enviroment.get_reward(action))

test()