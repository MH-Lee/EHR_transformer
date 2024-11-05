import torch
import torch.nn as nn
from torch import Tensor


class BalancedBinaryCrossEntropyLoss(nn.Module):
    def __init__(
            self, alpha: float = None, device: str = "cpu"
    ):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.device = device
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, y: Tensor):
        bce_loss = self.bce_loss_fn(logits, y)
        weight = self.get_weight(y)

        return torch.mean(weight * bce_loss)

    def get_weight(self, y: Tensor):
        if self.alpha is None:
            return torch.ones_like(y)
        else:
            return self.alpha * y + (1 - self.alpha) * (1 - y)
        

class FocalLoss(nn.Module):
    # TODO: implement focal loss. if alpha is not None, use balanced focal loss.
    def __init__(self, gamma: float, alpha: float = None, device: str = "cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

    def forward(self, logits: Tensor, y: Tensor):
        y_prob = torch.sigmoid(logits)
        fl = (
                - y * torch.pow(1 - y_prob, self.gamma) * torch.log(y_prob) -
                (1 - y) * torch.pow(y_prob, self.gamma) * torch.log(1 - y_prob)
        )
        if self.alpha is not None:
            weight = self.get_weight(y)
            fl = weight * fl

        return torch.mean(fl)

    def get_weight(self, y: Tensor):
        if self.alpha is None:
            return torch.ones_like(y)
        else:
            return self.alpha * y + (1 - self.alpha) * (1 - y)