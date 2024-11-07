import sys
import json
import os
import pickle
import random
import time
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from src.loader import PretrainedEHRDataset, collate_fn_pt
from src.pretrained_models import BERT
from src.pretrained_models.train import train_model, evaluate_model
from src.loss import BalancedBinaryCrossEntropyLoss, FocalLoss
from src.pretrained_models.utils import get_datasets, get_logger
from pprint import pprint
from datetime import datetime
import os.path as osp
import wandb
from easydict import EasyDict as edict

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True) 


class Runner:
    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.logger = get_logger(args.log_name, args.log_dir, args.config_dir)
        self.logger.info(vars(args))
        pprint(vars(args))
        if self.args.device != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(args.device)
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        
        self.seed = args.seed
        self.logger.info(f'device: {self.device}')
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.code_idx = pickle.load(open(osp.join(args.data_dir, 'code_indices', 'code_dict_pretrained.pkl'), 'rb'))
        self.load_data(pretraine_type=args.pretrained_type, load_pretrained=args.use_pretrained)
        if self.args.use_pretrained:
            self.model = BERT(vocab_size=len(self.code_idx), pretrained_emb=self.gpt4o_emb.float(),
                              embed_dim=self.args.embed_dim, num_heads=self.args.num_heads,\
                              hidden_dim=self.args.ffn_dim, num_layers=self.args.num_layers, \
                              max_len=self.args.max_len, attn_dropout=self.args.attn_dropout, \
                              dropout_rate=self.args.dropout_rate, device=self.device, \
                              num_classes=self.args.num_classes).to(self.device)
        else:
            self.model = BERT(vocab_size=len(self.code_idx), embed_dim=self.args.embed_dim, \
                              num_heads=self.args.num_heads, hidden_dim=self.args.ffn_dim, \
                              num_layers=self.args.num_layers, max_len=self.args.max_len, \
                              attn_dropout=self.args.attn_dropout, dropout_rate=self.args.dropout_rate, \
                              device=self.device, num_classes=self.args.num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.mlm_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        if self.args.loss_type == 'bce':
            self.cls_criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss_type == 'balanced_bce':
            self.cls_criterion = BalancedBinaryCrossEntropyLoss(alpha=args.alpha, device=self.device)
        elif self.args.loss_type == 'focalloss':
            self.cls_criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha, device=self.device)
        else:
            raise NotImplementedError(f'{self.args.loss_type} is not implemented')
        
        
    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved
        
        Returns
        -------
        """
        state = {
            'state_dict'   : self.model.state_dict(),
            'best_val'     : self.best_val,
            'best_epoch'   : self.best_epoch,
            'optimizer'	   : self.optimizer.state_dict(),
            'args'		   : vars(self.args)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model
        
        Returns
        -------
        """
        state             = torch.load(load_path)
        state_dict		  = state['state_dict']
        self.best_val     = state['best_val']
        self.best_val_auc = self.best_val['auc'] 
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def load_data(self, pretraine_type='te3-small', load_pretrained=False):
        if osp.isfile(osp.join(self.args.data_dir, 'split_dataset_pt', f'train_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt')):
            train_datasets = torch.load(osp.join(self.args.data_dir, 'split_dataset_pt', f'train_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
            valid_datasets = torch.load(osp.join(self.args.data_dir, 'split_dataset_pt', f'valid_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
            test_datasets = torch.load(osp.join(self.args.data_dir, 'split_dataset_pt', f'test_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
            self.logger.info(f'train_ds: {len(train_datasets)} | valid_ds: {len(valid_datasets)} | test_ds: {len(test_datasets)}')
        else:
            train_ds, valid_ds, test_ds = get_datasets(osp.join(self.args.data_dir, 'split_datasets'), self.seed)
            self.logger.info(f'train_ds: {len(train_ds)} | valid_ds: {len(valid_ds)} | test_ds: {len(test_ds)}')
            train_datasets = PretrainedEHRDataset(train_ds, mask_prob=self.args.mask_prob, max_len=self.args.max_len)
            valid_datasets = PretrainedEHRDataset(valid_ds, mask_prob=self.args.mask_prob, max_len=self.args.max_len)
            test_datasets = PretrainedEHRDataset(test_ds, mask_prob=self.args.mask_prob, max_len=self.args.max_len)
            torch.save(train_datasets, osp.join(self.args.data_dir, 'split_dataset_pt', f'train_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
            torch.save(valid_datasets, osp.join(self.args.data_dir, 'split_dataset_pt', f'valid_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
            torch.save(test_datasets, osp.join(self.args.data_dir, 'split_dataset_pt', f'test_ds_seed{str(self.seed)}_mask{str(self.args.mask_prob)}.pt'))
        
        self.train_loader = DataLoader(train_datasets, batch_size=self.args.batch_size, shuffle=True,\
                                       num_workers=self.args.num_workers, collate_fn=collate_fn_pt)
        self.valid_loader = DataLoader(valid_datasets, batch_size=self.args.batch_size, shuffle=False, \
                                       num_workers=self.args.num_workers, collate_fn=collate_fn_pt)
        self.test_loader  = DataLoader(test_datasets, batch_size=self.args.batch_size, shuffle=False, \
                                       num_workers=self.args.num_workers, collate_fn=collate_fn_pt)

        if load_pretrained:
            if pretraine_type == 'te3-small':
                self.gpt4o_emb = pickle.load(open(osp.join(self.args.data_dir, 'gpt_emb', 'gpt4o_te3_small_v2.pkl'), 'rb'))
            elif pretraine_type == 'te3-large':
                self.gpt4o_emb = pickle.load(open(osp.join(self.args.data_dir, 'gpt_emb', 'gpt4o_te3_large_v2.pkl'), 'rb'))
            elif pretraine_type == 'te-ada002':
                self.gpt4o_emb = pickle.load(open(osp.join(self.args.data_dir, 'gpt_emb', 'gpt4o_te_ada002_v2.pkl'), 'rb'))
            else:
                raise NotImplementedError(f'{pretraine_type} is not implemented')
        
    def fit(self):
        tr_loss_list = list()
        val_loss_list = list()
        tr_mlm_loss_list = list()
        tr_cls_loss_list = list()
        val_mlm_loss_list = list()
        val_cls_loss_list = list()
        counter = 0
        self.best_val = 0.0
        self.best_epoch = 0
        model_save_path = ''
        
        for epoch in range(self.args.max_epoch):
            train_log = train_model(model=self.model, 
                                    loader=self.train_loader,
                                    optimizer=self.optimizer, 
                                    mlm_criterion=self.mlm_criterion, 
                                    cls_criterion=self.cls_criterion,
                                    epoch=epoch, device=self.device, 
                                    logger=self.logger, 
                                    wandb_logger=self.writer,
                                    diag_freeze=self.args.diag_freeze)
            valid_log = evaluate_model(model=self.model, 
                                       loader=self.valid_loader,
                                       mlm_criterion=self.mlm_criterion, 
                                       cls_criterion=self.cls_criterion, 
                                       epoch=epoch, 
                                       device=self.device, 
                                       logger=self.logger,
                                       wandb_logger=self.writer,
                                       mode='valid')
            tr_loss_list.append(train_log['loss'])
            tr_mlm_loss_list.append(train_log['mlm_loss'])
            tr_cls_loss_list.append(train_log['cls_loss'])
            val_loss_list.append(valid_log['loss'])
            val_mlm_loss_list.append(valid_log['mlm_loss'])
            val_cls_loss_list.append(valid_log['cls_loss'])
            
            current_score = valid_log['auc']
            if current_score > self.best_val:
                if osp.isfile(model_save_path):
                    os.remove(model_save_path)
                self.best_epoch = epoch
                self.best_val = valid_log['auc']
                counter = 0
                model_filename = self.args.name + f'_best_epoch:{self.best_epoch}.pt'
                model_save_path = os.path.join(self.args.checkpoint_dir, self.date_str, model_filename)                
                self.save_model(model_save_path)
                # torch.save(self.model.state_dict(), f'{model_save_path}')
            else:
                counter += 1
                self.logger.info(f"Early stopping counter: {counter}/{self.args.patience}")
              
            if counter >= self.args.patience:
                self.logger.info(f"Early stopping triggered at epoch {self.best_epoch}")
                self.logger.info(f"Best Combined Score (AUC): {self.best_val:.4f}")
                break
        
        self.load_data(model_save_path)
        test_log = evaluate_model(model=self.model, 
                                  loader=self.valid_loader,
                                  mlm_criterion=self.mlm_criterion, 
                                  cls_criterion=self.cls_criterion, 
                                  epoch=epoch, 
                                  device=self.device, 
                                  logger=self.logger,
                                  wandb_logger=self.writer,
                                  mode='test')
        with open(os.path.join(self.args.log_dir, model_filename + f'_best_epoch:{self.best_epoch}.txt'), 'w') as f:
            f.write('\n')
            f.write(str(test_log))
            f.write('\n')
        return test_log['acc'], test_log['precision'], test_log['recall'], test_log['f1'], test_log['auc']
    
    
def main():
    with wandb.init() as writer:
        # import pdb; pdb.set_trace()
        torch.manual_seed(wandb.config.seed)
        torch.cuda.manual_seed(wandb.config.seed)
        torch.cuda.manual_seed_all(wandb.config.seed)
        np.random.seed(wandb.config.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(wandb.config.seed)
        date_dir = datetime.today().strftime("%Y%m%d")
        log_dir = osp.join('./src/pretrained_models/log/', date_dir)

        if wandb.config.use_pretrained:
            if wandb.config.diag_freeze:            
                wandb_name = f'EHR-Transformer_{wandb.config.seed}_diag_freeze_GPT4O_SWEEPS'
                tags = (str(wandb.config.seed), 'diag_freeze', 'te3-small')
            else:
                wandb_name = f'EHR-Transformer_{wandb.config.seed}_GPT4O_SWEEPS'
                tags = (str(wandb.config.seed,), 'te3-small')
        else:
            if wandb.config.diag_freeze:
                wandb_name = f'EHR-Transformer_{wandb.config.seed}_diag_freeze_SWEEPS'
                tags = (str(wandb.config.seed), 'diag_freeze')
            else:
                wandb_name = f'EHR-Transformer_{wandb.config.seed}_SWEEPS'
                tags = (str(wandb.config.seed),)

        name = f'{wandb_name}_{str(date_dir)}-{str(time.strftime("%H:%M:%S"))}' 
        log_name = f'{name}.log'
        writer.name = f'{wandb_name}_{writer.config.loss_type}_{str(date_dir)}'
        writer.tags =  writer.tags + tags

        if wandb.config.use_pretrained:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')

        args_dict = {
            'model_name': 'EHR-Transformer',
            'data_dir': './new_data/',
            'config_dir': './src/config/',
            'checkpoint_dir': './src/pretrained_models/results/',
            'name': name,
            'log_dir': log_dir,
            'log_name': log_name,
            'device': device,
            'max_epoch': wandb.config.max_epoch,
            'max_len': 400,
            'num_classes': 100,
            'num_workers': 16,
            'embed_dim': wandb.config.embed_dim,
            'batch_size': 256,
            'lr': wandb.config.learning_rate,
            'patience': 5,
            'mask_prob': wandb.config.mask_prob,
            'num_heads': wandb.config.num_heads,
            'ffn_dim': wandb.config.dim_feedforward,
            'num_layers': wandb.config.num_layers,
            'dropout_rate': wandb.config.dropout_rate,
            'attn_dropout': wandb.config.attention_dropout,
            'loss_type': wandb.config.loss_type,
            'alpha': None,
            'gamma': wandb.config.gamma,
            'use_pretrained': wandb.config.use_pretrained,
            'pretrained_type': 'te3-small',
            'diag_freeze': wandb.config.diag_freeze,
            'seed': wandb.config.seed,
        }

        args = edict(args_dict)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(osp.join(args.checkpoint_dir, date_dir), exist_ok=True)
        model = Runner(args, writer)
        acc, prec, rec, f1, auc = model.fit()
    
    

if __name__ == '__main__':
    sweep_configuration = {
        'method': 'bayes',  # 검색 방법 (옵션: 'random', 'grid', 'bayes')
        'metric': {'name': 'valid AUC', 'goal': 'maximize'},
        'parameters': {
            'max_epoch': {'values': [100]},
            'learning_rate': {'values': [5e-4, 1e-4, 5e-4, 1e-3]},
            'seed': {'values': [123, 321, 666, 777, 5959]},
            'embed_dim' : {'values': [128, 256, 512, 768]},
            'num_heads' : {'values': [2, 4, 8]},
            'dim_feedforward': {'values': [1024, 2048, 4096]},
            'num_layers': {'values': [2, 4, 6]},
            'dropout_rate': {'values': [0.1, 0.2, 0.3]},
            'attention_dropout': {'values': [0.1, 0.2, 0.3]},
            'mask_prob': {'values': [0.15, 0.2, 0.25, 0.3]},
            'loss_type': {'values': ['bce', 'balanced_bce', 'focalloss']},
            'use_pretrained': {'values': [True, False]},
            'diag_freeze': {'values': [True, False]},
            'gamma': {'values': [0.5, 1.0, 2.0]},
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, entity='leemh', project="EHR-Project-New")
    wandb.agent(sweep_id, function=main, count=100)

    
    
