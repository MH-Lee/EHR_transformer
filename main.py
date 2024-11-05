import os 
import pickle
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
import numpy as np
import os.path as osp
import time
import warnings
from pprint import pprint
from torch.utils.data import DataLoader

from src.models import TransformerTimeGPT
from src.loss import BalancedBinaryCrossEntropyLoss, FocalLoss
from src.models.train import train_model, evaluate_model
from src.loader import EHRDataset, collate_fn
from models.utils import get_logger, get_indices, get_data


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(args.name, args.log_dir, args.config_dir)
        self.logger.info(vars(args))
        pprint(vars(args))
        if self.args.device != 'cpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        
        self.seed = args.seed
        self.logger.info(f'device: {self.device}')
        data_dict = pickle.load(open(osp.join(args.data_dir, 'data_dict_preprocess_maxlen50.pkl'), 'rb'))
        self.indices_dir = osp.join(self.args.data_dir, 'split_indices')
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.load_data(data_dict, pretraine_type=args.pretrained_type, load_pretrained=args.use_pretrained)
        
        if self.args.model_name.lower() == 'hitanet':
            self.code_idx = pickle.load(open(osp.join(args.data_dir, 'code_indices', 'code2idx.pkl'), 'rb'))
            # diag_size = len([key for key in self.code_idx.keys() if 'd_' in key]) + 1
            if self.args.use_pretrained:
                self.model = TransformerTimeGPT(n_codes=len(self.code_idx),
                                                pretrained_emb=self.gpt4o_emb.float(),
                                                batch_size=args.batch_size, 
                                                device=self.device,
                                                args=self.args).to(self.device)
            else:
                self.model = TransformerTimeGPT(n_codes=len(self.code_idx),
                                                batch_size=args.batch_size, 
                                                device=self.device,
                                                args=self.args).to(self.device)
        else:
            raise NotImplementedError(f'{self.args.model_name} is not implemented')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        if self.args.loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.args.loss_type == 'balanced_bce':
            self.loss = BalancedBinaryCrossEntropyLoss(alpha=args.alpha, device=self.device)
        elif self.args.loss_type == 'focalloss':
            self.loss = FocalLoss(gamma=args.gamma, alpha=args.alpha, device=self.device)
        else:
            raise NotImplementedError(f'{self.args.loss_type} is not implemented')


    def load_data(self, data_dict, pretraine_type='te3-small', load_pretrained=False):
        tr_indices, val_indices, te_indices = get_indices(self.indices_dir , self.seed)
        train_data, valid_data, test_data = get_data(data_dict, tr_indices, val_indices, te_indices)
        
        train_dataset = EHRDataset(train_data)
        valid_dataset = EHRDataset(valid_data)
        test_dataset = EHRDataset(test_data)
        
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
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
        counter = 0
        best_score = 0.0
        best_epoch = 0
        model_save_path = ''
        
        for epoch in tqdm(range(self.args.max_epoch)):
            if self.args.model_name.lower() == 'hitanet':
                train_log = train_model(model=self.model, loader=self.train_loader, optimizer=self.optimizer, criterion=self.loss, \
                                        epoch=epoch, device=self.device, model_name=self.args.model_name, logger=self.logger, args=self.args)
                valid_log = evaluate_model(model=self.model, loader=self.valid_loader, criterion=self.loss, epoch=epoch,
                                           device=self.device, logger=self.logger, model_name=self.args.model_name, mode='valid')
                tr_loss_list.append(train_log['loss'])
                val_loss_list.append(valid_log['loss'])
            else:
                raise NotImplementedError(f'{self.args.model_name} is not implemented')
                
            current_score = valid_log['auc']
            if current_score > best_score:
                if osp.isfile(model_save_path):
                    os.remove(model_save_path)
                best_epoch = epoch
                counter = 0
                best_score = valid_log['auc']
                model_filename = self.args.name  + f'_best_epoch:{best_epoch}.pt'
                model_save_path = os.path.join(self.args.checkpoint_dir, self.date_str, model_filename)
                torch.save(self.model.state_dict(), f'{model_save_path}')
            else:
                counter += 1
                self.logger.info(f"Early stopping counter: {counter}/{self.args.patience}")
              
            if counter >= self.args.patience:
                self.logger.info(f"Early stopping triggered at epoch {best_epoch}")
                self.logger.info(f"Best Combined Score (AUC): {best_score:.4f}")
                break
        
        if self.args.model_name.lower() == 'hitanet':
            test_log = evaluate_model(model=self.model, loader=self.valid_loader, criterion=self.loss, epoch=epoch,
                                      device=self.device, logger=self.logger, model_name=self.args.model_name, mode='test')
        else:
            raise NotImplementedError(f'{self.args.model_name} is not implemented')  
        with open(os.path.join(self.args.log_dir, model_filename + f'_best_epoch:{best_epoch}.txt'), 'w') as f:
            f.write('\n')
            f.write(str(test_log))
            f.write('\n')
        return test_log['acc'], test_log['precision'], test_log['recall'], test_log['f1'], test_log['auc']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EHR mimic-iv train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--log_dir', type=str, default='./src/models/log/', help='log directory')
    parser.add_argument('--config_dir', type=str, default='./src/config/', help='config directory')
    parser.add_argument('--model_name', type=str, default='hitanet', help='model name')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/', help='model directory')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--model_dim', type=int, default=256, help='model dimension')
    parser.add_argument('--ffn_dim', type=int, default=1024, help='feedforward net dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--loss_type', type=str, default='bce', help='loss type')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=None, help='alpha for balanced bce or focal loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma for focal loss')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--use_pretrained', action="store_true", help='use pretrained model')
    parser.add_argument('--pretrained_type', type=str, default='te3-small', help='pretrained model type')
    parser.add_argument('--diag_freeze', action="store_true", help='pretrained embedding freeze')
    args = parser.parse_args()
    
    seed_list = [123, 321, 666, 777, 5959]
    results = []
    date_dir = datetime.today().strftime("%Y%m%d")
    

    args.log_dir = osp.join(args.log_dir, date_dir)
    
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
                args.name = f'{args.model_name}_dim{args.model_dim}_nl{args.num_layers}_{args.pretrained_type}_freeze_loss_{args.loss_type}_{date_dir}' + time.strftime('%H:%M:%S') + '_SEED_'
            else:
                args.name = f'{args.model_name}_dim{args.model_dim}_nl{args.num_layers}_{args.pretrained_type}_loss_{args.loss_type}_{date_dir}_' + time.strftime('%H:%M:%S') + '_SEED_'
        else:
            args.name = f'{args.model_name}_dim{args.model_dim}_nl{args.num_layers}_loss_{args.loss_type}_{date_dir}_' + time.strftime('%H:%M:%S') + '_SEED_'
            
        args.seed = seed
        args.name = args.name + str(seed)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(osp.join(args.checkpoint_dir, date_dir), exist_ok=True)
        
        model = Runner(args)
        acc, prec, rec, f1, auc = model.fit()
        results.append([acc, prec, rec, f1, auc])

    log_file = f'./src/models/results/{args.name}.txt'
    results = np.array(results)
    print(np.mean(results, 0))
    with open(log_file, 'w') as f:
        f.write(args.model_name)
        f.write('\n')
        f.write(str(np.mean(results, 0)))
        f.write('\n')
        f.write(str(np.std(results, 0)))
    f.close()