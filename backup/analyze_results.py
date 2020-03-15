import argparse
from PIL import Image
import os, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from utils.evaluation import dice_score
# future-self: dice and f1 are the same thing, but if you use f1_score from sklearn it will be much slower, the reason
# being that dice here expects bools and it won't work in multi-class scenarios. Same goes for accuracy_score.
# (see https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DRIVE', help='which dataset to test')
parser.add_argument('--experiment_path', type=str, default=None, help='experiment to be evaluated')
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_list, mask_list, gt_list = df.im_paths, df.mask_paths, df.gt_paths

    all_bin_preds = []
    all_preds = []
    all_gts = []
    for i in range(len(gt_list)):
        im_path = im_list[i].rsplit('/', 1)[-1]
        pred_path = osp.join(path_to_preds, im_path[:-4] + '.npy')
        bin_pred_path = osp.join(path_to_preds, im_path[:-4] + '_binary.png')
        gt_path = gt_list[i]
        mask_path = mask_list[i]

        gt = np.array(Image.open(gt_path)).astype(bool)
        mask = np.array(Image.open(mask_path).convert('L')).astype(bool)


        try: pred = np.load(pred_path)
        except FileNotFoundError:
            sys.exit('---- no predictions found (maybe run first generate_results.py?) ---- '+ pred_path)
        # os.remove(pred_path)
        bin_pred = np.array(Image.open(bin_pred_path).convert('L')).astype(bool)

        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        pred_flat = pred.ravel()
        bin_pred_flat = bin_pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_pred = pred_flat[mask_flat == True]
        noFOV_bin_pred = bin_pred_flat[mask_flat == True]

        # accumulate gt pixels and prediction pixels
        all_preds.append(noFOV_pred)
        all_bin_preds.append(noFOV_bin_pred)
        all_gts.append(noFOV_gt)

    return np.hstack(all_preds), np.hstack(all_bin_preds), np.hstack(all_gts)

def cutoff_youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def cutoff_dice(preds, gts):
    dice_scores = []
    thresholds = np.linspace(0, 1, 1001)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds>thresh
        dice_scores.append(dice_score(gts, hard_preds))
    dices = np.array(dice_scores)
    optimal_threshold = thresholds[dices.argmax()]
    return optimal_threshold

def cutoff_accuracy(preds, gts):
    accuracy_scores = []
    thresholds = np.linspace(0, 1, 1001)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds > thresh
        accuracy_scores.append(accuracy_score(gts.astype(np.bool), hard_preds.astype(np.bool)))
    accuracies = np.array(accuracy_scores)
    optimal_threshold = thresholds[accuracies.argmax()]
    return optimal_threshold

def compute_performance(preds, bin_preds, gts, save_path=None, opt_threshold=None, cut_off='youden', mode='train'):
    fpr, tpr, thresholds = roc_curve(gts, preds)
    global_auc = auc(fpr, tpr)

    if save_path is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve')
        ll = 'AUC = {:4f}'.format(global_auc)
        plt.legend([ll], loc='lower right')
        fig.tight_layout()
        if opt_threshold is None:
            if mode=='train':
                plt.savefig(osp.join(save_path,'ROC_train.png'))
            elif mode=='val':
                plt.savefig(osp.join(save_path, 'ROC_val.png'))
        else:
            plt.savefig(osp.join(save_path, 'ROC_test.png'))

    if opt_threshold is None:
        if cut_off == 'acc':
            # this would be to get accuracy-maximizing threshold
            opt_threshold = cutoff_accuracy(preds, gts)
        elif cut_off == 'dice':
            # this would be to get dice-maximizing threshold
            opt_threshold = cutoff_dice(preds, gts)
        else:
            opt_threshold = cutoff_youden(fpr, tpr, thresholds)

    acc = accuracy_score(gts, preds > opt_threshold)
    dice = dice_score(gts, preds > opt_threshold)
    dice_from_bin = dice_score(gts, bin_preds)

    tn, fp, fn, tp = confusion_matrix(gts, preds > opt_threshold).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, dice, dice_from_bin, specificity, sensitivity, opt_threshold


if __name__ == '__main__':
    '''
    Example:
    python analyze_results.py --experiment_path unet_adrian --dataset DRIVE
    python analyze_results.py --experiment_path unet_adrian --dataset CHASEDB
    python analyze_results.py --experiment_path unet_adrian --dataset STARE
    python analyze_results.py --experiment_path unet_adrian --dataset AV-WIDE
    '''
    exp_path = 'experiments/'
    results_path = 'results/'

    # gather parser parameters
    args = parser.parse_args()
    dataset = args.dataset
    if args.experiment_path is None: raise Exception('must specify experiment_path, sorry')
    exp_name = args.experiment_path
    cut_off = args.cut_off

    # print('* Analyzing performance in ' + dataset + ' training set -- Retrieving optimal threshold')
    # path_to_preds = osp.join(results_path, dataset + '/experiments', exp_name)
    # save_path = osp.join(path_to_preds, 'perf')
    # perf_csv_path = osp.join(save_path, 'training_performance.csv')
    # if osp.exists(perf_csv_path):
    #     global_auc_tr, acc_tr, dice_tr, dice_from_bin_tr,\
    #     spec_tr, sens_tr, opt_thresh_tr = pd.read_csv(perf_csv_path).values[0]
    #     print('-- Performance in DRIVE training set had been pre-computed, optimal threshold = {:.4f}'.format(
    #         opt_thresh_tr))
    # else:
    #     preds, bin_preds, gts = get_labels_preds(path_to_preds, csv_path = 'data/' + dataset + '/test.csv')
    #     os.makedirs(save_path, exist_ok=True)
    #     global_auc_tr, acc_tr, dice_tr, dice_from_bin_tr,\
    #     spec_tr, sens_tr, opt_thresh_tr = compute_performance(preds, bin_preds, gts, save_path=save_path,
    #                                                       opt_threshold=None, cut_off=cut_off, mode='train')
    #     perf_df_train = pd.DataFrame({'auc': global_auc_tr,
    #                                   'acc': acc_tr,
    #                                   'dice/F1': dice_tr,
    #                                   'dice/F1_from_bin': dice_from_bin_tr,
    #                                   'spec': spec_tr,
    #                                   'sens': sens_tr,
    #                                   'opt_t': opt_thresh_tr}, index=[0])
    #     perf_df_train.to_csv(perf_csv_path, index=False)
    #
    # print('* Analyzing performance in DRIVE validation set -- Retrieving optimal threshold')
    # path_to_preds = osp.join(results_path, dataset, 'experiments', exp_name)
    # save_path = osp.join(path_to_preds, 'perf')
    # perf_csv_path = osp.join(save_path, 'validation_performance.csv')
    # if osp.exists(perf_csv_path):
    #     global_auc_vl, acc_vl, dice_vl, dice_from_bin_vl,\
    #     spec_vl, sens_vl, opt_thresh_vl = pd.read_csv(perf_csv_path).values[0]
    #     print('-- Performance in DRIVE validation set had been pre-computed, optimal threshold = {:.4f}'.format(
    #         opt_thresh_vl))
    # else:
    # preds, bin_preds, gts = get_labels_preds(path_to_preds, csv_path = 'data/' + dataset + '/test.csv')
    # os.makedirs(save_path, exist_ok=True)
    # global_auc_vl, acc_vl, dice_vl, dice_from_bin_vl,\
    # spec_vl, sens_vl, opt_thresh_vl = compute_performance(preds, bin_preds, gts, save_path=save_path,
    #                                                       opt_threshold=None, cut_off=cut_off, mode='val')
    # perf_df_val = pd.DataFrame({'auc': global_auc_vl,
    #                             'acc': acc_vl,
    #                             'dice/F1': dice_vl,
    #                             'dice/F1_from_bin': dice_from_bin_vl,
    #                             'spec': spec_vl,
    #                             'sens': sens_vl,
    #                             'opt_t': opt_thresh_vl}, index=[0])
    # perf_df_val.to_csv(perf_csv_path, index=False)
    # print('* Saving CSV to: ' + perf_csv_path)

    print('* Analyzing performance in ' + dataset + ' test set')
    path_to_preds = osp.join(results_path, dataset, 'experiments', exp_name)
    save_path = osp.join(path_to_preds, 'perf')
    os.makedirs(save_path, exist_ok=True)
    perf_csv_path = osp.join(save_path, 'test_performance.csv')
    path_test_csv = osp.join('data', dataset, 'test.csv')
    preds, bin_preds, gts = get_labels_preds(path_to_preds, csv_path = path_test_csv)
    global_auc_test, acc_test, dice_test, dice_from_bin_test,\
    spec_test, sens_test, _ = compute_performance(preds, bin_preds, gts, save_path=save_path)#, opt_threshold=opt_thresh_vl)
    perf_df_test = pd.DataFrame({'auc': global_auc_test,
                                 'acc': acc_test,
                                 'dice/F1': dice_test,
                                 'dice/F1_from_bin':dice_from_bin_test,
                                 'spec': spec_test,
                                 'sens': sens_test}, index=[0])
    perf_df_test.to_csv(perf_csv_path, index=False)
    print('* Done')
    print('AUC in Train/Val/Test set is {:.4f}'.format(global_auc_test))
    print('Accuracy in Train/Val/Test set is {:.4f}'.format( acc_test))
    print('Dice/F1 score in Train/Val/Test set is {:.4f}'.format( dice_test))
    print('Dice/F1 score **from binarized** in Train/Val/Test set is {:.4f}'.format(dice_from_bin_test))
    print('Specificity in Train/Val/Test set is {:.4f}'.format(spec_test))
    print('Sensitivity in Train/Val/Test set is {:.4f}'.format(sens_test))
    print('ROC curve plots saved to ', save_path)
