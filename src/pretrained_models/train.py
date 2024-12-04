import torch
import wandb
import numpy as np
from src.metrics import compute_average_accuracy, compute_average_auc, compute_average_f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(model, loader, optimizer, mlm_criterion, cls_criterion, epoch, device, logger, wandb_logger=None, \
                use_thresholds=True, diag_freeze=False, mlm_lambda=0.5, num_classes=100, mlm_loss_type='ce'):
    # 모델을 학습하기 위한 함수
    model.train()
    train_loss = 0.0
    train_mlm_loss = 0.0
    train_cls_loss = 0.0
    thresholds = None
    total_pred = torch.empty((0, num_classes), device=device)
    total_true = torch.empty((0, num_classes), device=device)
    
    for batch_data in tqdm(loader):
        optimizer.zero_grad()
        labels = batch_data['labels'].float().to(device)
        mlm_output, logits_cls = model(batch_data)
        mlm_tokens = batch_data['mask_tokens'].to(device)
        mlm_masks = (mlm_tokens != 0).long()
        # import pdb; pdb.set_trace()
        if mlm_loss_type == 'mse':
            selected_emb = model.code_emb(mlm_tokens)
            mlm_loss = mlm_criterion(mlm_output, selected_emb)
            mlm_loss = mlm_loss * mlm_masks.unsqueeze(2)
            # import pdb; pdb.set_trace()
        else:
            mlm_loss = mlm_criterion(mlm_output.transpose(1,2), mlm_tokens)
            
        mlm_loss_mean = (mlm_loss.sum() / mlm_masks.sum())
        cls_loss = cls_criterion(logits_cls, labels)
        
        pred_probas = torch.sigmoid(logits_cls)
        total_pred = torch.cat((total_pred, pred_probas), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)
        
        # loss = mlm_loss_mean + cls_loss
        loss = (mlm_lambda * mlm_loss_mean) + ((1-mlm_lambda) * cls_loss)

        loss.backward()
        if diag_freeze:
            # import pdb; pdb.set_trace()
            model.code_emb.weight.grad[0:864] = 0
        optimizer.step()
        
        train_mlm_loss += (mlm_loss_mean.item() * mlm_lambda)
        train_cls_loss += (cls_loss.item() * (1 - mlm_lambda))
        train_loss += loss.item()

    tr_avg_loss = train_loss / len(loader)
    tr_avg_mlm_loss = train_mlm_loss / len(loader)
    tr_avg_cls_loss = train_cls_loss / len(loader)

    f1_recall_prec = compute_average_f1_score(total_pred.cpu().detach(), 
                                              total_true.cpu().detach(), 
                                              reduction='macro', 
                                              thresholds=thresholds)
    f1 = f1_recall_prec['average_f1_score']
    precision = f1_recall_prec['average_precision']
    recall = f1_recall_prec['average_recall']

    if use_thresholds:
        thresholds = f1_recall_prec['thresholds']
    else:
        thresholds = None

    auc = compute_average_auc(total_pred.cpu().detach(), 
                              total_true.cpu().detach(),
                              reduction='mean')
    acc = compute_average_accuracy(total_pred.cpu().detach(), 
                                   total_true.cpu().detach(),
                                   reduction='mean',
                                   thresholds=thresholds)['accuracies']
    
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
def evaluate_model(model, loader, mlm_criterion, cls_criterion, epoch, device, logger, wandb_logger=None,\
                   test_class_name=None, use_thresholds=True, mode='valid', mlm_lambda=0.5, num_classes=100, mlm_loss_type='ce'):
    model.eval()
    val_loss = 0.0 
    val_mlm_loss = 0.0
    val_cls_loss = 0.0
    thresholds = None
    total_pred = torch.empty((0, num_classes), device=device)
    total_true = torch.empty((0, num_classes), device=device)
    
    for batch_data in loader:
        labels = batch_data['labels'].float().to(device)
        mlm_output, logits_cls = model(batch_data)
        mlm_tokens = batch_data['mask_tokens'].to(device)
        mlm_masks = (mlm_tokens != 0).long()
        
        if mlm_loss_type == 'mse':
            selected_emb = model.code_emb(mlm_tokens)
            mlm_loss = mlm_criterion(mlm_output, selected_emb)
            mlm_loss = mlm_loss * mlm_masks.unsqueeze(2)
        else:
            mlm_loss = mlm_criterion(mlm_output.transpose(1,2), mlm_tokens)
        
        mlm_loss_mean = (mlm_loss * mlm_masks).sum() / mlm_masks.sum()
        cls_loss = cls_criterion(logits_cls, labels)
        
        pred_probas = torch.sigmoid(logits_cls)
        total_pred = torch.cat((total_pred, pred_probas), dim=0)
        total_true = torch.cat((total_true, labels), dim=0)

        loss = (mlm_lambda * mlm_loss_mean) + ((1-mlm_lambda) * cls_loss)
        val_loss += loss.item()
        val_mlm_loss += (mlm_loss_mean.item() * mlm_lambda)
        val_cls_loss += (cls_loss.item() * (1 - mlm_lambda))
    
    val_avg_loss = val_loss / len(loader)
    val_avg_mlm_loss = val_mlm_loss / len(loader)
    val_avg_cls_loss = val_cls_loss / len(loader)
    y_true, y_pred_prob = total_true.cpu().detach(), total_pred.cpu().detach()
    
    if mode == 'valid':
        f1_recall_prec = compute_average_f1_score(y_pred_prob, y_true, reduction='macro', thresholds=thresholds)
        f1 = f1_recall_prec['average_f1_score']
        precision = f1_recall_prec['average_precision']
        recall = f1_recall_prec['average_recall']
        
        if use_thresholds:
            thresholds = f1_recall_prec['thresholds']
        else:
            thresholds = None        
        
        auc = compute_average_auc(y_pred_prob, y_true, reduction='mean')
        acc = compute_average_accuracy(y_pred_prob, y_true, reduction='mean', thresholds=thresholds)['accuracies']
        
    elif mode == 'test':
        test_class_name = test_class_name
        f1_recall_prec_raw = compute_average_f1_score(y_pred_prob, y_true, reduction='none', thresholds=thresholds)
        f1_raw = f1_recall_prec_raw['average_f1_score']
        prec_raw = f1_recall_prec_raw['average_precision']
        rec_raw = f1_recall_prec_raw['average_recall']
        
        if use_thresholds:
            thresholds = f1_recall_prec_raw['thresholds']
        else:
            thresholds = None
        
        auc_raw = compute_average_auc(y_pred_prob, y_true, reduction='none')
        acc_raw = compute_average_accuracy(y_pred_prob, y_true, reduction='none', thresholds=thresholds)['accuracies']
        
        acc_data  = [[label, val] for (label, val) in zip(test_class_name, list(acc_raw))]
        auc_data  = [[label, val] for (label, val) in zip(test_class_name, list(auc_raw))]
        f1_data   = [[label, val] for (label, val) in zip(test_class_name, list(f1_raw))]
        prec_data = [[label, val] for (label, val) in zip(test_class_name, list(prec_raw))]
        rec_data  = [[label, val] for (label, val) in zip(test_class_name, list(rec_raw))]
        
        acc_table = wandb.Table(data=acc_data, columns = ["Diagnosis_name", "Accuracy"])
        auc_table = wandb.Table(data=auc_data, columns = ["Diagnosis_name", "AUC"])
        f1_table = wandb.Table(data=f1_data, columns = ["Diagnosis_name", "F1"])
        prec_table = wandb.Table(data=prec_data, columns = ["Diagnosis_name", "Precision"])
        rec_table = wandb.Table(data=rec_data, columns = ["Diagnosis_name", "Recall"])

        f1 = round(float(np.mean(f1_raw)), ndigits=4)
        precision = round(float(np.mean(prec_raw)), ndigits=4)
        recall = round(float(np.mean(rec_raw)), ndigits=4)
        acc = round(float(np.mean(acc_raw)), ndigits=4)
        auc = round(float(np.mean(auc_raw)), ndigits=4)
        
        wandb.log({"ACC_Chart" : wandb.plot.bar(acc_table, "Diagnosis_name", "Accuracy", title="Accuracy per class")})
        wandb.log({"AUC_Chart" : wandb.plot.bar(auc_table, "Diagnosis_name", "AUC", title="AUC per class")})
        wandb.log({"F1_Chart" : wandb.plot.bar(f1_table, "Diagnosis_name", "F1", title="F1 per class")})
        wandb.log({"Precision_Chart" : wandb.plot.bar(prec_table, "Diagnosis_name", "Precision", title="Precision per class")})
        wandb.log({"Recall_Chart" : wandb.plot.bar(rec_table, "Diagnosis_name", "Recall", title="Recall per class")})
    else:
        raise ValueError('mode should be either valid or test')
    
    logger.info(f'[Epoch {mode} {epoch}]: Total loss: {val_avg_loss:.4f} | MLM loss: {val_avg_mlm_loss:.4f} | CLS loss: {val_avg_cls_loss:.4f}')
    logger.info(f'[Epoch {mode} {epoch}]: Acuuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if mode == 'valid':
        wandb_logger.log(data={f'{mode} Loss': val_avg_loss, f'{mode} MLM Loss': val_avg_mlm_loss, f'{mode} CLS Loss': val_avg_cls_loss}, step= epoch)
        wandb_logger.log(data={f'{mode} Accuracy': acc, f'{mode} AUC': auc, f'{mode} F1': f1, f'{mode} Precision': precision, f'{mode} Recall': recall}, step= epoch)
    else:
        wandb_logger.log(data={f'{mode} Accuracy': acc, f'{mode} AUC': auc, f'{mode} F1': f1, f'{mode} Precision': precision, f'{mode} Recall': recall})
        cm_plot_dict = {}
        roc_plot_dict = {}
        for i in tqdm(range(total_pred.size(1))):
            if thresholds is None:
                y_pred= y_pred_prob[:, i].ge(0.5).float().numpy()
                cm_fig, _ = plot_confusion_matrix(y_true[:, i].numpy(), y_pred, test_class_name[i])
                roc_fig, _ = plot_roc_curve(y_true[:, i].numpy(), y_pred_prob[:, i].numpy(), test_class_name[i])
                cm_plot_dict[f'ConfusionMatrix/Confusion matrix {test_class_name[i]}'] = cm_fig
                roc_plot_dict[f'ROC_plot/ROC {test_class_name[i]}'] = roc_fig
                plt.close(cm_fig)
                plt.close(roc_fig)
            else:
                y_pred = y_pred_prob[:, i].ge(thresholds[i]).float().numpy()
                cm_fig, ax = plot_confusion_matrix(y_true[:, i].numpy(), y_pred, test_class_name[i])
                roc_fig, ax = plot_roc_curve(y_true[:, i].numpy(), y_pred_prob[:, i].numpy(), test_class_name[i])
                cm_plot_dict[f'ConfusionMatrix/Confusion matrix {test_class_name[i]}'] = cm_fig
                roc_plot_dict[f'ROC_plot/ROC {test_class_name[i]}'] = roc_fig
                plt.close(cm_fig)
                plt.close(roc_fig)
                
        wandb_logger.log(cm_plot_dict)
        wandb_logger.log(roc_plot_dict)
        
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
    
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm_fig, ax = plt.subplots(figsize=(20, 16))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True'])
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix {class_names}", fontsize=16)
    return cm_fig, ax

def plot_roc_curve(y_true, y_score, class_names):
    roc_fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {class_names}')
    plt.legend(loc="lower right")
    return roc_fig, ax