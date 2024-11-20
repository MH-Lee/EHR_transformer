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
        
        self.labels_list = pickle.load(open(osp.join(args.data_dir, 'top_100_list.pkl'), 'rb'))
        
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
        self.best_epoch   = state['best_epoch']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])
        self.logger.info(f'Model loaded from {load_path}')

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
                                    use_thresholds=self.args.use_thresholds,
                                    diag_freeze=self.args.diag_freeze)
            valid_log = evaluate_model(model=self.model, 
                                       loader=self.valid_loader,
                                       mlm_criterion=self.mlm_criterion, 
                                       cls_criterion=self.cls_criterion, 
                                       epoch=epoch, 
                                       device=self.device, 
                                       logger=self.logger,
                                       use_thresholds=self.args.use_thresholds,
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
                model_save_path = os.path.join(self.args.checkpoint_dir, model_filename)                
                self.save_model(model_save_path)
                # torch.save(self.model.state_dict(), f'{model_save_path}')
            else:
                counter += 1
                self.logger.info(f"Early stopping counter: {counter}/{self.args.patience}")
              
            if counter >= self.args.patience:
                self.logger.info(f"Early stopping triggered at epoch {self.best_epoch}")
                self.logger.info(f"Best Combined Score (AUC): {self.best_val:.4f}")
                break
        
        self.load_model(model_save_path)
        test_log = evaluate_model(model=self.model, 
                                  loader=self.valid_loader,
                                  mlm_criterion=self.mlm_criterion, 
                                  cls_criterion=self.cls_criterion, 
                                  epoch=epoch, 
                                  device=self.device, 
                                  logger=self.logger,
                                  wandb_logger=self.writer,
                                  use_thresholds=self.args.use_thresholds,
                                  test_class_name=self.labels_list,
                                  mode='test')
        self.logger.info(f"Best Combined metrics: \n {str(test_log)}")
        return test_log['acc'], test_log['precision'], test_log['recall'], test_log['f1'], test_log['auc']
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EHR mimic-iv train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str, default='EHR-Transformer', help='model name')
    parser.add_argument('--data_dir', type=str, default='./new_data/', help='data directory')
    parser.add_argument('--expset_dir', type=str, default='./src/pretrained_models/exp_setting', help='experiment setting directory')
    parser.add_argument('--log_dir', type=str, default='./src/pretrained_models/log/', help='log directory')
    parser.add_argument('--config_dir', type=str, default='./src/config/', help='config directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./src/pretrained_models/results/', help='model directory')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--mask_prob', type=float, default=0.2, help='mask probability')
    parser.add_argument('--max_len', type=int, default=400, help='max length')
    parser.add_argument('--embed_dim', type=int, default=768, help='model dimension')
    parser.add_argument('--ffn_dim', type=int, default=2048, help='feedforward net dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--loss_type', type=str, default='bce', help='loss type')
    parser.add_argument('--attn_dropout', type=float, default=0.3, help='attention dropout ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=None, help='alpha for balanced bce or focal loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma for focal loss')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--use_pretrained', action="store_true", help='use pretrained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pretrained_type', type=str, default='te3-small', help='pretrained model type')
    parser.add_argument('--diag_freeze', action="store_true", help='pretrained embedding freeze')
    parser.add_argument('--use_wandb', action="store_true", help='use wandb (defalut: False)')
    parser.add_argument('--use_thresholds', action="store_true", help='use thresholds for evaluation')
    parser.add_argument('--exp_num', type=int, default=0, required=True, help='experiment number')
    parser.add_argument('--mlm_lambda', type=float, default=0.5, help='lambda for mlm loss')
    args = parser.parse_args()
    
    seed_list = [123, 321, 666, 777, 5959]
    results = []
    date_dir = datetime.today().strftime("%Y%m%d")
    
    if args.use_wandb:
        wandb_mode = 'online'
    else:
        wandb_mode = 'disabled'
    
    for seed in seed_list:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed)
        
        if args.use_pretrained:
            if args.diag_freeze:
                args.name = f'{args.model_name}-GPT4O-EXP{str(seed)}-{str(args.exp_num)}_{date_dir}' 
                wandb_name = f'{args.model_name}-GPT4O-EXP{str(seed)}-{str(args.exp_num)}'
                tags2 = 'diag_freeze'
                tags3 = args.pretrained_type
                tags4 = args.loss_type
            else:
                args.name = f'{args.model_name}-GPT4O_EXP{str(seed)}-{str(args.exp_num)}_{date_dir}'
                wandb_name = f'{args.model_name}-GPT4O_EXP{str(seed)}-{str(args.exp_num)}'
                tags2 = 'diag_freeze_not'
                tags3 = args.pretrained_type
                tags4 = args.loss_type
        else:
            if args.diag_freeze:
                args.name = f'{args.model_name}-EXP{str(seed)}-{str(args.exp_num)}-{date_dir}'
                wandb_name = f'{args.model_name}-EXP{str(seed)}-{str(args.exp_num)}'
                tags2 = 'diag_freeze'
                tags3 = 'no_pretrained'
                tags4 = args.loss_type
            else:
                args.name = f'{args.model_name}-EXP{str(seed)}-{str(args.exp_num)}-{date_dir}'
                wandb_name = f'{args.model_name}-EXP{str(seed)}-{str(args.exp_num)}'
                tags2 = 'diag_freeze_not'
                tags3 = 'no_pretrained'
                tags4 = args.loss_type
            
        wandb_config = {
            'model_name'        : args.model_name,
            'seed'              : seed,
            'device'            : args.device,
            'batch_size'        : args.batch_size,
            'max_epoch'         : args.max_epoch,
            'learning_rate'     : args.lr,
            'mask_prob'         : args.mask_prob,
            'attention_dropout' : args.attn_dropout,
            'dropout_rate'      : args.dropout_rate,
            'embed_dim'         : args.embed_dim,
            'num_heads'         : args.num_heads,
            'num_layers'        : args.num_layers,
            'dim_feedforward'   : args.ffn_dim,
            'num_classes'       : args.num_classes,
            'loss_type'         : args.loss_type,
            'alpha'             : args.alpha,
            'gamma'             : args.gamma,
            'patience'          : args.patience,
            'use_pretrained'    : args.use_pretrained,
            'pretrained_type'   : args.pretrained_type,
            'diag_freeze'       : args.diag_freeze,
            'mlm_lambda'        : args.mlm_lambda
        }
        
        writer = wandb.init(project='EHR-Project-New', name=wandb_name, config=wandb_config, tags=[str(seed), tags2, tags3, tags4],
                            reinit=True, settings=wandb.Settings(start_method='thread'), mode=wandb_mode)
        
        args.seed = seed
        args.name = args.name + str(seed)
        args.log_name = args.name + '.log'
        exp_setting = vars(args)
        os.makedirs(args.expset_dir, exist_ok=True)
        with open(osp.join(args.expset_dir, args.name + '.json'), 'w') as f:
            json.dump(exp_setting, f)
        f.close()
        
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        model = Runner(args, writer)
        wandb.watch(model.model, log='all')
        acc, prec, rec, f1, auc = model.fit()
        results.append([acc, prec, rec, f1, auc])
        writer.finish()

    log_file = f'./src/pretrained_models/results_summary/{args.name}.txt'
    results = np.array(results)
    print(np.mean(results, 0))
    with open(log_file, 'w') as f:
        f.write(args.model_name)
        f.write('\n')
        f.write(str(np.mean(results, 0)))
        f.write('\n')
        f.write(str(np.std(results, 0)))
    f.close()