import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import rankdata

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def accuracy_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    num_els = actual.size
    intersection = np.logical_and(actual, predicted)
    return float(intersection.sum()) / num_els

def fast_auc(actual, predicted):
    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def ewma(data, window=5):
    # exponetially-weighted moving averages
    data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def evaluate(logits, labels, n_classes, ignore_index = -100, fast=True):
    all_preds = []
    all_targets = []

    act = torch.sigmoid if n_classes==1 else torch.nn.Softmax(dim=0)

    for i in range(len(logits)):
        prediction = act(logits[i]).detach().cpu().numpy()[-1] # this takes last channel in multi-class, ok for 2-class
        target = labels[i].cpu().numpy()
        all_preds.append(prediction.ravel())
        all_targets.append(target.ravel())

    all_preds_np = np.hstack(all_preds).ravel()
    all_targets_np = np.hstack(all_targets).ravel()

    all_preds_np = all_preds_np[all_targets_np != ignore_index]
    all_targets_np = all_targets_np[all_targets_np!=ignore_index]

    if fast:
        return fast_auc(all_targets_np, all_preds_np)
    else:
        return roc_auc_score(all_targets_np, all_preds_np)