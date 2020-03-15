import sys, json, os, argparse
from shutil import copyfile
import os.path as osp
from datetime import datetime
import operator

from tqdm import trange
from skimage.filters import threshold_otsu as threshold
import torch
from models.res_unet_adrian import UNet as unet

from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate, ewma, dice_score
from utils.model_saving_loading import save_model, str2bool, load_model

import matplotlib.pyplot as plt
from PIL import Image


# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:
parser.add_argument('--layers', type=str, default='16/32/64/128/256', help='unet configuration (filters x layer)')
parser.add_argument('--n_classes', type=int, default=1, help='number of target classes (1)')
parser.add_argument('--in_channels', type=int, default=3, help='green or rgb (rgb)')
parser.add_argument('--up_mode', type=str, default='transp_conv', help='upsampling scheme (\'transp_conv\')')
parser.add_argument('--pool_mode', type=str, default='max_pool', help='downsampling scheme (\'max_pool\')')
parser.add_argument('--conv_bridge', type=str2bool, nargs='?', const=True, default=False, help='convolutional bridge (False)')
parser.add_argument('--shortcut', type=str2bool, nargs='?', const=True, default=False, help='shortcut connections (False)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--disable_transforms', type=str2bool, nargs='?', const=True, default=False,
                    help='whether to disable data augmentation at 3*patience//4 (False)')
parser.add_argument('--eps', type=float, default=1e-8, help='eps parameter in Adam optimizer')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--data_aug', type=str, default='all', help='data augmentation strategies (\'all\'')
parser.add_argument('--normalize', type=str, default='from_im_max',
                    help='normalize (from_im_max,from_im,from_dataset), meaning: '
                         '(1/max(im), [im-m(im)]/s(im), [im-m(dataset)]/s(dataset). (from_im_max))')
parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (auc/loss)')
parser.add_argument('--patience', type=int, default=40, help='epochs until early stopping (40)')
parser.add_argument('--scheduler_f', type=float, default=0,
                    help='decay factor in after 3/4 of patience epochs (default=0=no decay)')
parser.add_argument('--n_epochs', type=int, default=300, help='total max epochs (300)')
parser.add_argument('--data_path', type=str, default='data/DRIVE/', help='where the training data is')
parser.add_argument('--end2end', type=str2bool, nargs='?', const=True, default=False,
                    help='whether to run bunch_evaluation after training (False)')
parser.add_argument('--checkpoint_path', type=str, default='nothing', help='checkpoint to load')
parser.add_argument('--solo', action='store_true', default=False, help='train on only SOLO pair (img - pseudo label)')
parser.add_argument('--classic', action='store_true', default=False, help='classic training (on training set)')
parser.add_argument('--forced_exp_name', type=str, default=False, help='force the script to save the experiment in a specific folder')

def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

def run_one_epoch(loader, model, criterion, optimizer=None):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    logits_all, labels_all = [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()
            else:
                loss = criterion(logits, labels)  # CrossEntropyLoss()

            if train:  # only in training mode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logits_all.extend(logits)
            labels_all.extend(labels)

            # Compute running loss
            running_loss += loss.item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train:
                t.set_postfix(tr_loss="{:.4f}".format(float(run_loss)))
            else:
                t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    return logits_all, labels_all, run_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, optimizer, criterion, train_loader, val_loader,
          n_epochs, metric, patience, scheduler_f, dis_transf, exp_path):
    counter_since_checkpoint = 0
    tr_losses, tr_aucs, tr_dices, vl_losses, vl_aucs, vl_dices = [], [], [], [], [], []
    stats = {}
    is_better, best_monitoring_metric = compare_op(metric)

    for epoch in range(n_epochs):
        print('\n EPOCH: {:d}/{:d}'.format(epoch+1, n_epochs))
        # train one epoch
        train_logits, train_labels, train_loss = run_one_epoch(train_loader, model, criterion, optimizer)
        train_auc = evaluate(train_logits, train_labels, model.n_classes)
        # validate one epoch, note no optimizer is passed
        with torch.no_grad():
            val_logits, val_labels, val_loss = run_one_epoch(val_loader, model, criterion)
            val_auc = evaluate(val_logits, val_labels, model.n_classes)
            # val_dice = dice_score(np.asarray(val_labels), np.asarray(val_logits) > threshold(val_logits))
        print('Train/Val Loss: {:.4f}/{:.4f}  -- Train/Val AUC: {:.4f}/{:.4f} -- LR={:.4f}'.format(
                train_loss, val_loss, train_auc, val_auc, get_lr(optimizer)))

        # store performance for this epoch
        tr_losses.append(train_loss)
        tr_aucs.append(train_auc)
        vl_losses.append(val_loss)
        vl_aucs.append(val_auc)
        #  smooth val values with a moving average before comparing
        val_auc = ewma(vl_aucs)[-1]
        val_loss = ewma(vl_losses)[-1]

        # check if performance was better than anyone before and checkpoint if so
        if metric=='auc':
            monitoring_metric = val_auc
        elif metric=='loss':
            monitoring_metric = val_loss
        # elif metric=='dice':
            # monitering_metric = val_dice

        if is_better(monitoring_metric, best_monitoring_metric):
            print('Best (smoothed) val {} attained. {:.4f} --> {:.4f}'.format(
                metric, best_monitoring_metric, monitoring_metric))
            print(17*'-',' Checkpointing ', 17*'-')

            best_monitoring_metric = monitoring_metric
            stats['tr_losses'] = tr_losses
            stats['vl_losses'] = vl_losses
            stats['tr_aucs'] = tr_aucs
            stats['vl_aucs'] = vl_aucs

            save_model(exp_path, model, optimizer, stats)
            counter_since_checkpoint = 0  # reset patience
        else:
            counter_since_checkpoint += 1

        if scheduler_f != 0 and counter_since_checkpoint == 3*patience//4:
            reduce_lr(optimizer, epoch, factor=scheduler_f, verbose=False)
            print(8 * '-', ' Reducing LR now ', 8 * '-')

        if counter_since_checkpoint == 3*patience//4 and dis_transf:
            print(8*'-', ' Disabling data augmentation now ', 8*'-')
            train_loader.dataset.transforms = val_loader.dataset.transforms

        # early stopping if no improvement happened for `patience` epochs
        if counter_since_checkpoint == patience:
            print('\n Early stopping the training, trained for {:d} epochs'.format(epoch))
            del model
            torch.cuda.empty_cache()
            return


    del model
    torch.cuda.empty_cache()
    return

if __name__ == '__main__':
    '''
    Example:
    python train.py --layers 64/128/256/512/1024 --experiment_path unet_adrian
    python train.py --layers 8/16 --pool_mode strided_conv --shortcut --conv_bridge --experiment_path unet_adrian
    '''
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reproducibility
    import numpy as np
    import random
    seed_value = 0
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if device is 'cuda':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

    # gather parser parameters
    n_classes = args.n_classes
    in_channels = args.in_channels
    layers = args.layers.split('/')
    layers = list(map(int, layers))
    up_mode = args.up_mode
    pool_mode = args.pool_mode
    conv_bridge = str2bool(args.conv_bridge) # guild ai gives strings back
    shortcut = args.shortcut
    lr = args.lr
    eps = args.eps
    dis_transf = args.disable_transforms
    bs = args.batch_size
    data_aug = args.data_aug
    normalize = args.normalize
    if normalize=='from_image' or normalize=='from_dataset': raise NotImplementedError
    metric = args.metric
    patience = args.patience
    scheduler_f = args.scheduler_f
    n_epochs = args.n_epochs
    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    solo, classic = args.solo, args.classic

    if(solo):
        print('* --solo chosen')
        if(classic):
            print('* You cannot chose both --solo and --classic')
            exit()
    elif(classic):
        print('* --classic chosen')
    else:
        print('* You must chose --solo or --classic')
        exit()

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M%S")
    if(args.forced_exp_name == False):
        experiment_path=osp.join('experiments', date_time)
    else:
        experiment_path=osp.join('experiments', args.forced_exp_name)


    args.experiment_path = experiment_path
    os.makedirs(experiment_path, exist_ok=True)

    config_file_path = osp.join(experiment_path,'config.cfg')
    args.config_file_path = config_file_path
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    # copy config file over results/ for later convenience reading results
    # copyfile(src=config_file_path, dst=osp.join('results', date_time+'_config.cfg'))

    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    train_loader, val_loader = get_train_val_loaders(data_path, in_channels=in_channels, normalize=normalize,
                                                     batch_size=bs, aug=data_aug, solo=solo)

    print('* Instantiating a Unet model with config = '+str(layers))
    model = unet(in_c=in_channels, n_classes=n_classes, layers=layers,
                 up_mode=up_mode, pool_mode=pool_mode,
                 conv_bridge=conv_bridge, shortcut=shortcut).to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, eps=eps)

    if n_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
    # n_classes=2
    print('* Instantiating loss function', str(criterion))
    print('* Starting to train')

    print('-' * 10)

    if(checkpoint_path != 'nothing'):
        print('* Model ' + checkpoint_path + ' loaded')
        checkpoint_path = osp.join('experiments', checkpoint_path)
        model, stats = load_model(model, checkpoint_path)
        print('* The AUC loaded is: '+str(stats['vl_aucs'][-1]))
        jauf_aucs = stats['tr_aucs']
        print('* The model loaded trained on '+str(len(jauf_aucs))+' epochs')

    train(model, optimizer, criterion, train_loader, val_loader, n_epochs, metric,
                          patience, scheduler_f, dis_transf, experiment_path)

    print("Model saved in: " + experiment_path)

    # print("Best validation metric was: %f" % best_monitoring_metric)
    if args.end2end:
        cmd = 'python bunch_evaluation.py --config_file ' + config_file_path
        os.system(cmd)
        perf_csv_path = osp.join('results', date_time+'_performance_summary.csv')
        import pandas as pd
        perf_df = pd.read_csv(perf_csv_path)
        val_auc =  perf_df.iat[1,1]
        print("val_auc: %f" % val_auc)
        test_auc = perf_df.iat[2,1]
        print("test_auc: %f" % test_auc)
