# import ml_collections
from ml_collections import config_dict

def get_config():
  """Get the default hyperparameter configuration."""
  config = config_dict.ConfigDict()

  config.learning_rate = 0.1
  config.batch_size = 128
  config.num_epochs = 100
  config.molecule = 'A4B'
  config.poly_degree = 3
  config.n_layers = 2
  config.n_neurons = 128
#   config.features = (128,128,)
  config.ntr = 1000
  config.nval = 1000
  config.ntst = 2000
  return config


def metrics():
  return []