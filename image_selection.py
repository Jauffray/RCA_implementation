import sys, json, os, argparse
import torch
import os.path as osp
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--img_slc', type=str, default=False, help='the image number selected, should be from the test set')

if __name__ == '__main__':
    args = parser.parse_args()
    img_slc = args.img_slc

    if (img_slc == False):
        print("* You must enter a 2-digits image number (ex: 01)")
        exit()

    # add other datasets than DRIVE later?

    # merge the CSVs
    DRIVE_path = 'data/DRIVE'
    train_path = osp.join(DRIVE_path, 'train.csv')
    val_path = osp.join(DRIVE_path, 'val.csv')
    test_path = osp.join(DRIVE_path, 'test.csv')

    dftr = pd.read_csv(train_path)
    dfv = pd.read_csv(val_path)
    dfte = pd.read_csv(test_path)

    imtr, masktr, gttr = dftr.im_paths, dftr.mask_paths, dftr.gt_paths
    imv, maskv, gtv = dfv.im_paths, dfv.mask_paths, dfv.gt_paths
    imte, maskte, gtte = dfte.im_paths, dfte.mask_paths, dfte.gt_paths

    im_list = np.hstack((dftr.im_paths,dfv.im_paths,dfte.im_paths))
    mask_list = np.hstack((dftr.mask_paths,dfv.mask_paths,dfte.mask_paths))
    gt_list = np.hstack((dftr.gt_paths,dfv.gt_paths,dfte.gt_paths))

    k = 0
    im_nbs = []
    for k in range(len(im_list)):
        im_nb = im_list[k].split('_')[0]
        im_nb = im_nb.split('\\')[-1]
        im_nbs.append(im_nb)

    idx = im_nbs.index(img_slc) # automatically return an error if not in the list

    im_solo = im_list[idx]
    print('* Found ' + im_solo)

    im_except_one = np.hstack((im_list[:idx], im_list[idx+1:]))
    mask_solo = mask_list[idx]
    mask_except_one = np.hstack((mask_list[:idx], mask_list[idx+1:]))
    gt_except_one = np.hstack((gt_list[:idx], gt_list[idx+1:]))

    # ext = gt_list[idx].split('.')[-1]

    # we don't use "ext" anymore because saving the pseudo_label as .gif instead
    # of .png was the cause of the inverted results
    gt_solo = gt_list[idx].split('.')[0] + '_pseudo.' + 'png'

    solo_df = pd.DataFrame({'im_paths': im_solo,
                            'gt_paths': gt_solo,
                            'mask_paths': mask_solo}, index = [0])

    except_one_df = pd.DataFrame({'im_paths': im_except_one,
                                'gt_paths': gt_except_one,
                                'mask_paths': mask_except_one})

    solo_csv = osp.join(DRIVE_path, 'solo.csv')
    except_one_csv = osp.join(DRIVE_path, 'except_one.csv')
    # if not osp.exists(perf_imgs_path):
    #     os.makedirs(perf_imgs_path)
    solo_df.to_csv(solo_csv, index=False)
    except_one_df.to_csv(except_one_csv, index=False)

    print('* Creating ' + solo_csv)
    print('* Creating ' + except_one_csv)
