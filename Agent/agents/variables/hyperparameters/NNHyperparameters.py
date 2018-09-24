class  NNHyperparameters():
	def __init__(self):
		print('NNHyperparameters inited!')

	def is_filled(self):
		isFilled = True
		members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
		for var in members:
			if getattr(self, var) is None:
				isFilled = False
		return isFilled