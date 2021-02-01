import numpy as np

#Need to fix the shit out of this config file - never use it and its making stuff messy af

config = dict(
    width = 100,
    height = 100,
    runtime = 300,
    threshold = 0.25,
    sigmoid_strength = 25,
    coupling = 0.55,
    refractory_period = 10,
    set_seed = 0, #0 if not setting, put number if you want to call a seed.
    constant = False, #
    gradient = False, #
    grad_start = 0.3, #
    grad_end = 0.4, #
    normal_modes = True,  #A, amp, mean
    normal_modes_config = [1, 0.1  ,0.9], #
    graph = True ,
    FullStateSave = 'full',      #Options r full (whole run), transition (one beat_period before AF, one beat_period after AF), False (Nothing saved)
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