import numpy as np

config = dict(
    width = 50,
    height = 50,
    runtime = 5000,
    threshold = 0.3,
    sigmoid_strength = 25,
    coupling = 0.55,
    refractory_period = 10,
    seed = 0,
    set_seed = False,
    constant = False,
    gradient = False,
    grad_start = 0.7,
    grad_end = 0.3,
    normal_modes = True,  #A1, A2,  B1,  B2, C1, C2,  alph,  beta
    normal_modes_config = [1  ,1  ,0.25  ,1  ,0  ,0  ,-0.1  ,0.7], 
    graph = False ,
    FullStateSave = 'full'      #Options r full (whole run), any number (last x timesteps), transition (150 before AF, 150 after AF), False (Nothign saved)
 )

if not config['set_seed']:
    config['seed'] = np.random.randint(0,int(1e7))

if config['gradient']:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Gradient" + "," + str(config['grad_start'])+ "," + str(config['grad_end']) +
      "," + str(config['refractory_period']) + ","+ str(config['seed']))
elif config['normal_modes']:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Normal Modes" + "," + "," + str(config['refractory_period']) + ","+ str(config['seed']))
else:
    title = (str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) +
     "," + str(config['sigmoid_strength']) + "," + "Isotropic" + str(config['coupling']) + "," + str(config['refractory_period']) +
      ","+ str(config['seed']))