import sys
import json
import pickle
import os.path as osp
import logging, logging.config

def get_datasets(path, seed):
    tr_ds = pickle.load(open(osp.join(path, f'train_datasets_{seed}.pkl'), 'rb'))
    val_ds = pickle.load(open(osp.join(path, f'valid_datasets_{seed}.pkl'), 'rb'))
    te_ds  = pickle.load(open(osp.join(path, f'test_datasets_{seed}.pkl'),  'rb'))
    return tr_ds, val_ds, te_ds

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open(config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + '/' + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)
	return logger
