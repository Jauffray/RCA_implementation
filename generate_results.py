import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm

import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.filters import threshold_otsu as threshold

import torch
from utils.model_saving_loading import str2bool
from models.res_unet_adrian import UNet as unet
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model

from PIL import Image


# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
required_named.add_argument('--dataset', type=str, help='which dataset to test', default='DRIVE')#required=True)

parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--solo', action='store_true', default=False, help='generate prediction on Solo set, store it as pseudo-label')
# parser.add_argument('--except_one', action='store_true', default=False, help='generate predictions on Except_one set')
parser.add_argument('--train', action='store_true', default=False, help='generate predictions on training set')

parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--binarize', type=str, default='otsu', help='binarization scheme (\'otsu\')')
parser.add_argument('--config_file', type=str, default=None, help='experiments/name_of_config_file, overrides everything')
# in case no config file is passed
parser.add_argument('--layers', type=str, default='16/32/64/128/256', help='unet configuration (filters x layer)')
parser.add_argument('--n_classes', type=int, default=1, help='number of target classes (1)')
parser.add_argument('--in_channels', type=int, default=3, help='green or rgb (rgb)')
parser.add_argument('--up_mode', type=str, default='transp_conv', help='upsampling scheme (\'transp_conv\')')
parser.add_argument('--pool_mode', type=str, default='max_pool', help='downsampling scheme (\'max_pool\')')
parser.add_argument('--conv_bridge', type=str2bool, nargs='?', const=True, default=False, help='convolutional bridge (False)')
parser.add_argument('--shortcut', type=str2bool, nargs='?', const=True, default=False, help='shortcut connections (False)')
parser.add_argument('--normalize', type=str, default='from_im_max',
                    help='normalize (from_im_max,from_im,from_dataset), meaning: '
                         '(1/max(im), [im-m(im)]/s(im), [im-m(dataset)]/s(dataset). (from_im_max))')
parser.add_argument('--forced_threshold', type=float, default=False, help='set the threshold of the binarized segmentation(s)')

def flip_ud(tens):
    return torch.flip(tens, dims=[1])


def flip_lr(tens):
    return torch.flip(tens, dims=[2])


def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])

def create_pred(model, tens, mask, coords_crop, original_sz, tta='no', binarize='otsu'):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)

    if tta!='no':
        with torch.no_grad():
            logits_lr = model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            logits_ud = model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            logits_lrud = model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1).flip(-2)

        if tta == 'from_logits':
            mean_logits = torch.mean(torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0)
            pred = act(mean_logits)
        elif tta == 'from_preds':
            pred_lr = act(logits_lr)
            pred_ud = act(logits_ud)
            pred_lrud = act(logits_lrud)
            pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)
        else: raise NotImplementedError
    pred = pred.detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class

    pred = resize(pred, output_shape=original_sz)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0

    if binarize=='otsu': from skimage.filters import threshold_otsu as threshold
    # may implement other binarization schemes, see skimage.filters
    else: raise NotImplementedError
    return full_pred, full_pred > threshold(full_pred)

def save_pred(preds, save_results_path, im_name, forced_threshold = False):
    full_pred, bin_pred = preds
    os.makedirs(save_results_path, exist_ok=True)
    im_name = im_name.rsplit('/', 1)[-1]
    # jauffray: handle the special case of DRIVE where there is a backslash instead of a normal slash
    im_name = im_name.rsplit('\\', 1)[-1]


    # img = Image.fromarray(np.uint8((full_pred > 0.91) * 255) , 'L')
    # img.show()
    # exit()


    if(solo ==  False):
        save_name = osp.join(save_results_path, im_name[:-4] + '.jpg')
        save_name_np = osp.join(save_results_path, im_name[:-4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # this casts preds to int, loses precision
            # we only do this for visualization purposes
            imsave(save_name, img_as_ubyte(full_pred))
        # we save float predictions in a numpy array for
        # accurate performance evaluation
        np.save(save_name_np, full_pred)
        # save also binarized image
        save_name = osp.join(save_results_path, im_name[:-4] + '_binary.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(save_name, img_as_ubyte(bin_pred))
    else:
        # save also binarized image
        # we force .png instead of copying the .gif extensions from other manuals because it caused the inverted image when loaded in train
        save_name = osp.join(save_results_path, im_name[:-4]+ '.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if(forced_threshold):
                imsave(save_name, img_as_ubyte(full_pred > forced_threshold))
            else:
                imsave(save_name, img_as_ubyte(bin_pred))

if __name__ == '__main__':
    '''
    Example:
    python generate_results.py --config_file whatever.cfg --dataset DRIVE --binarize otsu --tta from_preds --train
    or specify by hand (check defaults):
    python generate_results.py --layers 4/8/16 --experiment_path experiments/unet_adrian --dataset DRIVE --tta --train
    - available datasets = DRIVE (--train, --val, test), CHASEDB, STARE, AV-WIDE
    '''
    results_path = 'results/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    dataset = args.dataset
    binarize = args.binarize
    # train, val = args.train, args.val
    tta = args.tta
    solo, train = args.solo, args.train
    # except_one = args.except_one
    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)

    layers = args.layers
    layers = args.layers.split('/')
    layers = list(map(int, layers))
    n_classes = args.n_classes
    in_channels = args.in_channels
    up_mode = args.up_mode
    pool_mode = args.pool_mode
    conv_bridge = args.conv_bridge
    shortcut = args.shortcut
    normalize = args.normalize
    experiment_path = args.experiment_path # this should exist in a config file
    if experiment_path is None: raise Exception('must specify path to experiment')
    forced_threshold = args.forced_threshold

    # if train and dataset != 'DRIVE':
    #     raise Exception('For generating predictions on training set please use --dataset DRIVE')
    # if val and dataset != 'DRIVE':
    #     raise Exception('For generating predictions on validation set please use --dataset DRIVE')
    data_path = osp.join('data', dataset)

    if solo:
        csv_path = 'solo.csv'
        if(train):
            print('* You cannot chose both --solo and --except_one')
            exit()
    # elif except_one:
    #     csv_path = 'except_one.csv'
    elif train:
        csv_path = 'train.csv'
    else:
        print("* You must chose between --solo and --test (not --except_one anymore)")
        exit()


    print('* Reading test data from ' + osp.join(data_path, csv_path))
    test_dataset = get_test_dataset(data_path, csv_path=csv_path, in_channels=in_channels,
                                    normalize=normalize, tg_size=(512, 512))
    print('* Instantiating a Unet model with config = ' + str(layers))
    model = unet(in_c=in_channels, n_classes=n_classes, layers=layers,
                 up_mode=up_mode, pool_mode=pool_mode,
                 conv_bridge=conv_bridge, shortcut=shortcut).to(device)
    print('* Loading trained weights from ' + experiment_path)
    experiment_path = osp.join('experiments', experiment_path)
    try:
        model, stats = load_model(model, experiment_path)
    except RuntimeError:
        sys.exit('---- bad config specification (check layers, pool_mode, etc.) ---- ')
    model.eval()

    save_results_path = osp.join(results_path, dataset, experiment_path)
    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta, binarize=binarize)
        if solo:
            save_results_path = osp.join(data_path, 'manual')
            ext = im_name.split('.')[-1]
            im_name = im_name.split('_')[0] + '_manual1_pseudo.' + ext
        save_pred(full_pred, save_results_path, im_name, forced_threshold)

    print('* Saving predictions to ' + save_results_path)
    print('* Done')
