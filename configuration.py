import numpy as np

config = dict(width = 50,
    height = 50,
    runtime = 200,
    threshold = 0.2,
    sigmoid_strength = 5,
    coupling = 0.7,
    refractory_period = 10,
    seed = 0,
    set_seed = False,
    constant = False,
    gradient = True,
    grad_start = 0.9,
    grad_end = 0.1,
    normal_modes = False,
    graph = False ,
    FullStateSave = True
 )

if not config['set_seed']:
    config['seed'] = np.random.randint(0,int(1e7))

#Do more conditional titles i.e. for different coupling mechs

title = str(config['width']) + "," + str(config['height']) + "," + str(config['runtime']) + "," + str(config['threshold']) + "," + str(config['sigmoid_strength']) + "," + str(config['coupling']) + "," + str(config['refractory_period']) + ","+ str(config['seed'])