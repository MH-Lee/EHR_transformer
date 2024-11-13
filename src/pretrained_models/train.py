import torch
from src.metrics import compute_average_accuracy, compute_average_auc, compute_average_f1_score
from tqdm import tqdm

def train_model(model, loader, optimizer, mlm_criterion, cls_criterion, epoch, device, logger, wandb_logger, diag_freeze=False, mlm_lambda=0.5):
    # 모델을 학습하기 위한 함수
    model.train()
    train_loss = 0.0
    train_mlm_loss = 0.0
    train_cls_loss = 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in tqdm(loader):
        optimizer.zero_grad()
        labels = batch_data['labels'].float().to(device)
        mlm_output, logits_cls = model(batch_data)
        mlm_tokens = batch_data['mask_tokens'].to(device)
        mlm_masks = (mlm_tokens != 0).long()
        # import pdb; pdb.set_trace()
        mlm_loss = mlm_criterion(mlm_output.transpose(1,2), mlm_tokens)
        mlm_loss_mean = (mlm_loss.sum() / mlm_masks.sum())
        cls_loss = cls_criterion(logits_cls, labels)
        
        y_pred = torch.sigmoid(logits_cls)
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        
        # loss = mlm_loss_mean + cls_loss
        loss = (mlm_lambda * mlm_loss_mean) + ((1-mlm_lambda) * cls_loss)

        loss.backward()
        if diag_freeze:
            # import pdb; pdb.set_trace()
            model.code_emb.weight.grad[0:870] = 0
        optimizer.step()
        
        train_mlm_loss += (mlm_loss_mean.item() * mlm_lambda)
        train_cls_loss += (cls_loss.item() * (1 - mlm_lambda))
        train_loss += loss.item()

    tr_avg_loss = train_loss / len(loader)
    tr_avg_mlm_loss = train_mlm_loss / len(loader)
    tr_avg_cls_loss = train_cls_loss / len(loader)
    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                   total_true.cpu().detach(), 
                                   reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                              total_true.cpu().detach(), 
                              reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                  total_true.cpu().detach(), 
                                  reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    
    logger.info(f'[Epoch train {epoch}]: Total loss: {tr_avg_loss:.4f} | MLM loss: {tr_avg_mlm_loss:.4f} | CLS loss: {tr_avg_cls_loss:.4f}')
    logger.info(f'[Epoch train {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    wandb_logger.log(data={'Train Loss': tr_avg_loss, 'Train MLM Loss': tr_avg_mlm_loss, 'Train CLS Loss': tr_avg_cls_loss}, step= epoch)
    wandb_logger.log(data={'Train Accuracy': acc, 'Train AUC': auc, 'Train F1': f1, 'Train Precision': precision, 'Train Recall': recall}, step= epoch)
    return {
            'loss': tr_avg_loss, 
            'mlm_loss': tr_avg_mlm_loss, 
            'cls_loss': tr_avg_cls_loss
            }


@torch.no_grad()
def evaluate_model(model, loader, mlm_criterion, cls_criterion, epoch, device, logger, wandb_logger=None, mode='valid', mlm_lambda=0.5):
    model.eval()
    val_loss = 0.0 
    val_mlm_loss = 0.0
    val_cls_loss = 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        labels = batch_data['labels'].float().to(device)
        mlm_output, logits_cls = model(batch_data)
        mlm_tokens = batch_data['mask_tokens'].to(device)
        mlm_masks = (mlm_tokens != 0).long()
        mlm_loss = mlm_criterion(mlm_output.transpose(1,2), mlm_tokens)
        mlm_loss_mean = (mlm_loss * mlm_masks).sum() / mlm_masks.sum()
        cls_loss = cls_criterion(logits_cls, labels)
        
        y_pred = torch.sigmoid(logits_cls)
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)

        loss = (mlm_lambda * mlm_loss_mean) + ((1-mlm_lambda) * cls_loss)
        val_loss += loss.item()
        val_mlm_loss += (mlm_loss_mean.item() * mlm_lambda)
        val_cls_loss += (cls_loss.item() * (1 - mlm_lambda))
    
    val_avg_loss = val_loss / len(loader)
    val_avg_mlm_loss = val_mlm_loss / len(loader)
    val_avg_cls_loss = val_cls_loss / len(loader)
    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                   total_true.cpu().detach(), 
                                   reduction='mean')['accuracies']
    auc = compute_average_auc(total_pred.cpu().detach(), 
                              total_true.cpu().detach(), 
                              reduction='mean')
    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                              total_true.cpu().detach(), 
                                              reduction='macro')
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']
    logger.info(f'[Epoch {mode} {epoch}]: Total loss: {val_avg_loss:.4f} | MLM loss: {val_avg_mlm_loss:.4f} | CLS loss: {val_avg_cls_loss:.4f}')
    logger.info(f'[Epoch {mode} {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if mode == 'valid':
        wandb_logger.log(data={f'{mode} Loss': val_avg_loss, f'{mode} MLM Loss': val_avg_mlm_loss, f'{mode} CLS Loss': val_avg_cls_loss}, step= epoch)
        wandb_logger.log(data={f'{mode} Accuracy': acc, f'{mode} AUC': auc, f'{mode} F1': f1, f'{mode} Precision': precision, f'{mode} Recall': recall}, step= epoch)
    else:
        wandb_logger.log(data={f'{mode} Accuracy': acc, f'{mode} AUC': auc, f'{mode} F1': f1, f'{mode} Precision': precision, f'{mode} Recall': recall})
    return {
            'loss': val_avg_loss, 
            'mlm_loss': val_avg_mlm_loss,
            'cls_loss': val_avg_cls_loss,
            'acc': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
            }
    