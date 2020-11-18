import numpy as np

config = dict(
    width = 50,
    height = 50,
    runtime = 1000,
    threshold = 0.2,
    sigmoid_strength = 25,
    coupling = 0.55,
    refractory_period = 10,
    seed = 0,
    set_seed = False,
    constant = True,
    gradient = False,
    grad_start = 0.7,
    grad_end = 0.3,
    normal_modes = False,
    graph = False ,
    FullStateSave = True
 )

if not config['set_seed']:
    config['seed'] = np.random.randint(0,int(1e7))

#Do more conditional titles i.e. for different coupling mechs

title = str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) + "," + str(config['sigmoid_strength']) + "," + str(config['coupling']) + "," + str(config['refractory_period']) + ","+ str(config['seed'])