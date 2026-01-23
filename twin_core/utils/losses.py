import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1) + self.eps

        dice = (2 * intersection) / denom
        return 1 - dice.mean()
