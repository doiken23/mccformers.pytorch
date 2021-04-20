import torch
from torch import Tensor


def compute_accuracy(pred: Tensor, gt: Tensor, ignore: int = 0):
    """
    pred (torch.Tensor): predicted words shape of [L, N]
    gt (torch.Tensor): GT words shape of [L, N]
    ignore (int): ignored label
    """
    mask = gt != ignore
    tp = torch.logical_and(pred == gt, mask)

    return tp.sum() / mask.sum()
