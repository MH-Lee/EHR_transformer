import torch
from src.metrics import compute_average_accuracy, compute_average_auc, compute_average_f1_score


def train_model(model, loader, optimizer, criterion, epoch, device, model_name, logger, args):
    # 모델을 학습하기 위한 함수
    model.train()
    train_loss = 0.0
    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        optimizer.zero_grad()
        labels = batch_data['labels'].float().to(device)
        if model_name.lower() == 'hitanet':
            pred, attn_w = model(batch_data)
        loss = criterion(pred, labels)
        
        y_pred = torch.sigmoid(pred)
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        
        loss.backward()
        
        if args.diag_freeze:
            # import pdb; pdb.set_trace()
            model.feature_encoder.pre_embedding.weight.grad[0:870] = 0
        
        optimizer.step()
        train_loss += loss.item()

    tr_avg_loss = train_loss / len(loader)
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
    
    logger.info(f'[Epoch train {epoch}]: total loss: {tr_avg_loss:.4f}')
    logger.info(f'[Epoch train {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return {'loss': tr_avg_loss}


@torch.no_grad()
def evaluate_model(model, loader, criterion, epoch, device, logger, model_name='HiTANet', mode='valid'):
    model.eval()
    val_loss = 0.0    
    total_pred = torch.empty((0, 100), device=device)
    total_true = torch.empty((0, 100), device=device)
    
    for batch_data in loader:
        labels = batch_data['labels'].float().to(device)
        if model_name.lower() == 'hitanet':
            pred, attn_w = model(batch_data)
        loss = criterion(pred, labels)
        
        y_pred = torch.sigmoid(pred)
        total_pred = torch.cat((total_pred, y_pred), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        val_loss += loss.item()
    
    avg_val_loss = val_loss / len(loader)
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
    logger.info(f'[Epoch {mode} {epoch}]: total loss: {avg_val_loss:.4f}')
    logger.info(f'[Epoch {mode} {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return {
            'loss': avg_val_loss, 
            'acc': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
            }
    