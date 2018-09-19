
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):
	"""Wrapper class for model hyperparameters."""
	def __init__(self):
		self.num_layers = 2
		self.hidden_size = 128
		self.model = "lstm"
		

class DataCofig(object):
	"""Wrapper class for DataCofig hyperparameters."""
	def __init__(self):
		self.minlen = 4
		self.maxlen = 15
		self.cantoneses = "/home/pingan_ai/dxq/project/cantonese.txt"
		self.blessing_file = "/home/pingan_ai/dxq/project/gen_blessing/dataset/data/line_lyrics.txt"
		self.id2word = "/home/pingan_ai/dxq/project/blessing_generate/line_lyrics2word_re.json"
		self.word2id = "/home/pingan_ai/dxq/project/blessing_generate/line_lyrics2id_re.json"
		
class TrainingConfig(object):
	"""Wrapper class for TrainingConfig hyperparameters."""
	def __init__(self):
		self.bach_size = 256
		
class TestingConfig(object):
	"""Wrapper class for TestingConfig hyperparameters."""
	def __init__(self):
		self.bach_size = 1
		self.checkpoint_path = "/media/pingan_ai/dxq/gen_blessing/new_model/"
		# self.checkpoint_path = "/media/pingan_ai/dxq/gen_blessing/models_1/"