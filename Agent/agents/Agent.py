class Agent():
	def __init__(self):
		print('Agent inited!')

	def check_structure(self, structure):
		if not structure.is_filled():
			raise Exception('Some members of passed structure are not initialized')