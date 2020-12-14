import numpy as np

config = dict(
    width = 50,
    height = 50,
    runtime = 20000,
    threshold = 0.25,
    sigmoid_strength = 25,
    coupling = 0.55,
    refractory_period = 10,
    set_seed = 238207504, #0 if not setting, put number if you want to call a seed.
    constant = False,
    gradient = False,
    grad_start = 0.3,
    grad_end = 0.4,
    normal_modes = True,  #A1,  A2, amp, mean
    normal_modes_config = [0.25, 1, 0.1  ,0.9],
    graph = True ,
    FullStateSave = 'transition',      #Options r full (whole run), transition (one beat_period before AF, one beat_period after AF), False (Nothing saved)
    stats = False
 )


if config['gradient']:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Gradient" + "," + str(config['grad_start'])+ "," + str(config['grad_end']) +
      "," + str(config['refractory_period']) + ",")
elif config['normal_modes']:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Normal Modes" + "," + str(config['refractory_period']) + "," )
else:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Isotropic" + str(config['coupling']) + "," + str(config['refractory_period']) +
      ",")