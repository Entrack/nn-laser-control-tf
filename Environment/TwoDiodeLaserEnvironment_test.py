from environments.TwoDiodeLaserEnvironment import TwoDiodeLaserEnvironment as test_environment

def test():
	print('testing')
	enviroment = test_environment()
	print(enviroment.step(1))

test()