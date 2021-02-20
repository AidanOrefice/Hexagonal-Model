import numpy as np

#Need to fix the shit out of this config file - never use it and its making stuff messy af

config = dict(
    width = 100,
    height = 100,
    runtime = 10000,
    threshold = 0.25,
    sigmoid_strength = 25,
    coupling = 0.55,
    refractory_period = 10,
    set_seed = 0, #0 if not setting, put number if you want to call a seed.
    normal_modes_config = [1, 0.1  ,0.9], #
    graph = False ,
    FullStateSave = False,      #Options r full (whole run), transition (one beat_period before AF, one beat_period after AF), False (Nothing saved)
    stats = False
 )

title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Normal Modes" + "," + str(config['refractory_period']) + "," )
