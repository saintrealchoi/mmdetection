import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

import numpy as np

@weighted_loss
def nllloss(pred, target):
    """nll loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    # assert pred.size() == target.size() and target.numel() > 0
    pred_sigma = pred[...,-4:]
    pred_sigma = torch.sigmoid(pred_sigma)
    pred = pred[...,:-4]
    sigma_const = 3e-1
    epsilon = 1e-9
    loss_xy = - torch.log(
                torch.exp(- ((pred[..., :2] - target[..., :2]) ** 2.0 ) / (pred_sigma[..., :2] ** 2.0 ) / 2.0) / (torch.sqrt(2.0 * np.pi * (pred_sigma[..., :2]**2)) + sigma_const) + epsilon)
    loss_wh = - torch.log(
                torch.exp(- ((pred[..., 2:4] - target[..., 2:4]) ** 2.0) / (pred_sigma[..., 2:4] ** 2.0 ) / 2.0) / (torch.sqrt(2.0 * np.pi * (pred_sigma[..., 2:4]**2)) + sigma_const) + epsilon)
    loss = loss_xy+loss_wh
    return loss

@LOSSES.register_module()
class NLLLoss(nn.Module):
    """NLL loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(NLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * nllloss(
            pred, target,  weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox