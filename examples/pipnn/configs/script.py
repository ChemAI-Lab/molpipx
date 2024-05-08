from absl import app
from absl import flags

from ml_collections import config_flags
# from ml_collections import config_dict

_CONFIG = config_flags.DEFINE_config_file('config')
# _MY_FLAG = flags.DEFINE_integer('my_flag', None)

def main(_):
  print(_CONFIG.value)
#   print(_MY_FLAG.value)

if __name__ == '__main__':
  app.run(main)