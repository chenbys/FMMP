from cape_extloc.cfg.s1 import *

data['samples_per_gpu'] = 4
data['workers_per_gpu'] = 4

lr_config['step'] = [160, 180]
total_epochs = 200
checkpoint_config['interval'] = 200
