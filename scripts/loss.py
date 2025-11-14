import torch
import torch.nn as nn
from torch.nn import functional as F

class MCC_Loss(nn.Module):
    """
    Calculate Matthews Correlation Coefficient-based loss.
    Adapted from
    https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py
    https://arxiv.org/pdf/2010.13454.pdf

    Args:
        inputs (torch.Tensor): probability predictions
        targets (torch.Tensor): 0 or 1 ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        
        """
        Modify torch.add and add 1e-8 to prevent a nan derivative for sqrt(0)
        February 28, 2024 (Serena Chen, chens@ornl.gov)
        """
        denominator = torch.sqrt(
            torch.add(tp, fp, alpha=1)
            * torch.add(tp, fn, alpha=1)
            * torch.add(tn, fp, alpha=1)
            * torch.add(tn, fn, alpha=1) + 1e-8
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        
        return 1.0 - mcc