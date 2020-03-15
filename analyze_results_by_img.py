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
import pickle

# future-self: dice and f1 are the same thing, but if you use f1_score from sklearn it will be much slower, the reason
# being that dice here expects bools and it won't work in multi-class scenarios. Same goes for accuracy_score.
# (see https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DRIVE', help='which dataset to test')
parser.add_argument('--experiment_path', type=str, default=None, help='experiment to be evaluated')
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')
parser.add_argument('--solo', action='store_true', default=False, help='analyze the results of SOLO (with "hidden" GT)')
parser.add_argument('--train', action='store_true', default=False, help='analyze the results of the training set')

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_list, mask_list, gt_list = df.im_paths, df.mask_paths, df.gt_paths

    all_bin_preds = []
    all_preds = []
    all_gts = []
    for i in range(len(gt_list)):
        im_path = im_list[i].rsplit('/', 1)[-1]
        im_path = im_list[i].rsplit('\\', 1)[-1]

        if(solo):
            bin_pred_path = gt_list[i]
            gt_path = gt_list[i].rsplit('_', 1)[0] + '.' + 'gif'
        else:
            pred_path = osp.join(path_to_preds, im_path[:-4] + '.npy')
            bin_pred_path = osp.join(path_to_preds, im_path[:-4] + '_binary.png')
            gt_path = gt_list[i]
        mask_path = mask_list[i]

        gt = np.array(Image.open(gt_path)).astype(bool)
        mask = np.array(Image.open(mask_path).convert('L')).astype(bool)

        # os.remove(pred_path)
        bin_pred = np.array(Image.open(bin_pred_path).convert('L')).astype(bool)
        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        bin_pred_flat = bin_pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_bin_pred = bin_pred_flat[mask_flat == True]
        # accumulate gt pixels and prediction pixels
        all_bin_preds.append(noFOV_bin_pred)
        all_gts.append(noFOV_gt)

        if(solo == False):
            try: pred = np.load(pred_path)
            except FileNotFoundError:
                sys.exit('---- no predictions found (maybe run first generate_results.py?) ---- '+ pred_path)
            pred_flat = pred.ravel()
            noFOV_pred = pred_flat[mask_flat == True]
            all_preds.append(noFOV_pred)

    return all_preds, all_bin_preds, all_gts

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
            plt.savefig(osp.join(save_path, 'ROC.png'))

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
    solo, train = args.solo, args.train

    if args.experiment_path is None:
        if(solo == False):
            raise Exception('must specify experiment_path, sorry')
    exp_name = args.experiment_path
    cut_off = args.cut_off

    if (solo):
        exp_name = "last_solo"

    print('* Analyzing performance in ' + dataset + ' test set')
    path_to_preds = osp.join(results_path, dataset, 'experiments', exp_name)
    save_path = osp.join(path_to_preds, 'perf')
    os.makedirs(save_path, exist_ok=True)
    perf_imgs_path = osp.join(save_path, 'test_performance')

    if solo:
        chosen_csv = 'solo.csv'
        save_path = osp.join(path_to_preds, 'perf_SOLO')
        if(train):
            print('* You cannot chose both --solo and --train')
            exit()
    elif train:
        chosen_csv = 'train.csv'
        save_path = osp.join(path_to_preds, 'perf_TRAIN')
    else:
        print('* You must chose between --solo and --train')
        exit()

    path_csv = osp.join('data', dataset, chosen_csv)
    df = pd.read_csv(path_csv)
    im_list = df.im_paths
    # np.hstack can be useful if merging datasets is necessary

    preds, bin_preds, gts = get_labels_preds(path_to_preds, csv_path = path_csv)
    best_img_name = ''
    best_img_val = 0.0
    best_bin_img_name = ''
    best_bin_img_val = 0.0
    k = 0

    if(solo):
        preds = bin_preds
    for k in range(len(bin_preds)):
            im_path = im_list[k].rsplit('/', 1)[-1] #we only take what's after the last "/"
            im_path = im_list[k].rsplit('\\', 1)[-1] #handle the special case of "\" in the name instead of "/"
            im_name = im_path[:-4] #we delete the extension in the name

            global_auc_test, acc_test, dice_test, dice_from_bin_test,\
            spec_test, sens_test, _ = compute_performance(preds[k], bin_preds[k], gts[k], save_path=None)

            perf_df_test = pd.DataFrame({'auc': global_auc_test,
                                         'acc': acc_test,
                                         'dice/F1': dice_test,
                                         'dice/F1_from_bin':dice_from_bin_test,
                                         'spec': spec_test,
                                         'sens': sens_test}, index=[0])
            perf_image = osp.join(save_path, im_name + '.csv')

            if not osp.exists(save_path):
                os.makedirs(save_path)
            perf_df_test.to_csv(perf_image, index=False)

            if(dice_from_bin_test > best_bin_img_val):
                best_bin_img_val = dice_from_bin_test
                best_bin_img_name = im_name

            if(dice_test > best_img_val):
                best_img_val = dice_test
                best_img_name = im_name

            print('* Treating '+ im_name)
            print('AUC is {:.4f}'.format(global_auc_test))
            print('Accuracy is {:.4f}'.format( acc_test))
            print('Dice/F1 score is {:.4f}'.format( dice_test))
            print('Dice/F1 score **from binarized** is {:.4f}'.format(dice_from_bin_test))
            print('Specificity is {:.4f}'.format(spec_test))
            print('Sensitivity is {:.4f}'.format(sens_test))

    print('* '+ best_img_name + ' get the best segmentation result, with a F1 score of ' + str(best_img_val))
    print('* '+ best_bin_img_name + ' get the best binarized segmentation result, with a (binarized) F1 score of ' + str(best_bin_img_val))

    pickle_list = [0.,0.,'']
    if(solo):
        pickle_list[0] = best_bin_img_val
        with open('objs.pkl', 'wb') as f:
            pickle.dump(pickle_list, f)
    if(train):
        pickle_list[1], pickle_list[2] = best_bin_img_val, best_bin_img_name
        with open('objs.pkl', 'wb') as f:
            pickle.dump(pickle_list, f)
