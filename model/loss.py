import torch
import torch.nn as nn
from .soft_skeleton import SoftSkeletonize


def soft_dice(y_true, y_pred):
    smooth = 1.0
    intersection = torch.sum(y_true * y_pred)
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1.0 - coeff


class improved_soft_dice_cldice(nn.Module):
    def __init__(self, alpha=0.5, beta=1, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.bce = nn.BCELoss()

    def forward(self, y_true, y_pred):
        dice = soft_dice(y_true, y_pred)

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        bce = self.bce(y_pred, y_true)

        return (1 - self.alpha) * dice + self.alpha * cl_dice + self.beta * bce