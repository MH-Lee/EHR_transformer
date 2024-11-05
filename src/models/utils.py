import sys
import pickle
import json
import os.path as osp
import logging, logging.config

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

def get_indices(path, seed):
    train_indices = pickle.load(open(osp.join(path, f'train_indices_{seed}.pkl'), 'rb'))
    valid_indices = pickle.load(open(osp.join(path, f'valid_indices_{seed}.pkl'), 'rb'))
    test_indices  = pickle.load(open(osp.join(path, f'test_indices_{seed}.pkl'),  'rb'))
    return train_indices, valid_indices, test_indices

def get_data(data, train_indices, valid_indices, test_indices):
    train_split_data = dict()
    valid_split_data = dict()
    test_split_data = dict()
    for p_id in train_indices:
        train_split_data[p_id] = data[p_id]
    for p_id in valid_indices:
        valid_split_data[p_id] = data[p_id]
    for p_id in test_indices:
        test_split_data[p_id] = data[p_id]
    return train_split_data, valid_split_data, test_split_data
