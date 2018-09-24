import tensorflow as tf

from agents.variables.structures.NNStructure import NNStructure

class BasicReinNNStructure(NNStructure):
	def __init__(self, *args, **kwargs):
		super(BasicReinNNStructure, self).__init__(*args, **kwargs)
		tf.reset_default_graph()
		self.inputs1 = None
		self.W = None
		self.Qout = None
		self.predict = None
		self.nextQ = None
		self.loss = None
		self.trainer = None
		self.updateModel = None
		print('BasicReinNNStructure inited!')