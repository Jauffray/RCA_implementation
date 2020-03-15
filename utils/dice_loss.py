# source : https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def binary_focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25, reduction="mean", function=torch.sigmoid, **kwargs):
    """
    Binary Version of Focal Loss
    :args

    y_pred : prediction

    y_true : true target labels

    gamma: dampeing factor default value 2 works well according to reasearch paper

    alpha : postive to negative ratio default value 0.25 means 1 positive and 3 negative can be tuple ,list ,int and float

    reduction = mean,sum,none

    function = can be sigmoid or softmax or None

    **kwargs: parameters to pass in activation function like dim in softmax

    """
    if isinstance(alpha, (list, tuple)):
        pos_alpha = alpha[0]  # postive sample ratio in the entire dataset
        neg_alpha = alpha[1]  # (1-alpha) # negative ratio in the entire dataset
    elif isinstance(alpha, (int, float)):
        pos_alpha = alpha
        neg_alpha = (1 - alpha)

    # if else in function can be simplified be removing setting to default to sigmoid  for educational purpose
    if function is not None:
        y_pred = function(y_pred, **kwargs)  # apply activation function
    else:
        assert ((y_pred <= 1) & (
            y_pred >= 0)).all().item(), "negative value in y_pred value should be in the range of 0 to 1 inclusive"

    pos_pt = torch.where(y_true == 1, y_pred, torch.ones_like(
        y_pred))  # positive pt (fill all the 0 place in y_true with 1 so (1-pt)=0 and log(pt)=0.0) where pt is 1
    neg_pt = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))  # negative pt

    pos_modulating = (
                         1 - pos_pt) ** gamma  # compute postive modulating factor for correct classification the value approaches to zero
    neg_modulating = (neg_pt) ** gamma  # compute negative modulating factor

    pos = -pos_alpha * pos_modulating * torch.log(pos_pt)  # pos part
    neg = -neg_alpha * neg_modulating * torch.log(1 - neg_pt)  # neg part

    loss = pos + neg  # this is final loss to be returned with some reduction

    # apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss  # reduction mean
    else:
        raise f
        "Wrong reduction {reduction} is choosen \n choose one among [mean,sum,none]  "

