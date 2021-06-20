import torch
import torch.nn as nn

def dice_coeff(preds, target):
    smooth = 1.
    # preds = (preds-preds.min())/(preds.max()-preds.min())
    num = preds.size(0)
    m1 = preds.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



 
 
class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight, size_average)
 
    def forward(self, preds, targets):
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(preds_flat, targets_flat)
    

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, preds, targets):
        num = targets.size(0)
        smooth = 1.
        
        probs = preds
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    
    
# 针对二分类任务的 Focal Loss
class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha,1-alpha]).cuda()
        self.gamma = gamma
        self.bce_loss = BCELoss2d()
        self.dice_loss = SoftDiceLoss()
 
 
    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        # loss = loss_bce
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.view(-1))
        pt = torch.exp(-loss)
        F_loss = at*(1-pt)**self.gamma * loss
        return F_loss.mean()