import tensorflow as tf

from agents.variables.structures.NNStructure import NNStructure

class ContextualBanditReinNNStructure(NNStructure):
	def __init__(self, *args, **kwargs):
		super(ContextualBanditReinNNStructure, self).__init__(*args, **kwargs)
		tf.reset_default_graph()
		
		
		
		print('ContextualBanditReinNNStructure inited!')