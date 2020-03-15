import os, sys, argparse, json
import pandas as pd
import os.path as osp
from utils.model_saving_loading import str2bool

# argument parsing
parser = argparse.ArgumentParser()

# parser.add_argument('--tta', action='store_true', default=False, help='test time augmentation (True)')
parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--binarize', type=str, default='otsu', help='binarization scheme (\'otsu\')')
parser.add_argument('--config_file', type=str, default=None, help='experiments/name_of_config_file, overrides everything')
# in case no config file is passed
parser.add_argument('--layers', type=str, default='64/128/256/512/1024', help='unet configuration (filters x layer)')
parser.add_argument('--n_classes', type=int, default=1, help='number of target classes (1)')
parser.add_argument('--in_channels', type=int, default=3, help='green or rgb (rgb)')
parser.add_argument('--up_mode', type=str, default='transp_conv', help='upsampling scheme (\'transp_conv\')')
parser.add_argument('--pool_mode', type=str, default='max_pool', help='downsampling scheme (\'max_pool\')')
parser.add_argument('--conv_bridge', type=str2bool, nargs='?', const=True, default=False, help='convolutional bridge (False)')
parser.add_argument('--shortcut', type=str2bool, nargs='?', const=True, default=False, help='shortcut connections (False)')
parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=False, help='Normalize by mean/std (False)')
parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None) # required
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')

if __name__ == '__main__':
    '''
    Example:
    python bunch_evaluation.py --config_file whatever.cfg --binarize otsu
    '''
    args = parser.parse_args()
    tta = args.tta
    binarize = args.binarize
    cut_off = args.cut_off
    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)

    layers = args.layers
    n_classes = args.n_classes
    in_channels = args.in_channels
    up_mode = args.up_mode
    pool_mode = args.pool_mode
    conv_bridge = args.conv_bridge
    shortcut = args.shortcut
    normalize = args.normalize
    experiment_path = args.experiment_path # this should exist in a config file
    if experiment_path is None: raise Exception('experiment_path, sorry')

    # build base string of flags
    base_cmd = 'python generate_results.py --layers ' + layers \
               + ' --n_classes ' + str(n_classes) \
               + ' --in_channels ' + str(in_channels) \
               + ' --up_mode ' + str(up_mode) \
               + ' --pool_mode ' + str(pool_mode) \
               + ' --conv_bridge ' + str(conv_bridge) \
               + ' --shortcut ' + str(shortcut) \
               + ' --normalize ' + str(normalize) \
               + ' --experiment_path ' + str(experiment_path) \
               + ' --tta ' + tta \
               + ' --binarize ' + binarize


    data_sets = ['DRIVE', 'STARE', 'CHASEDB', 'AV-WIDE']
    data_set = 'DRIVE'
    cmd_1 = base_cmd + ' --dataset ' + data_set+ ' --train'
    cmd_2 = base_cmd + ' --dataset ' + data_set+ ' --val'

    try:
        os.system(cmd_1) # need to fix this try/except, os.system() always succeeds
        os.system(cmd_2)  # need to fix this try/except, os.system() always succeeds
    except:
        sys.exit('Something went wrong, exiting to avoid entering a loop of exceptions')

    data_sets = ['DRIVE', 'STARE', 'CHASEDB', 'AV-WIDE']
    for data_set in data_sets:
        cmd = base_cmd + ' --dataset ' + data_set
        os.system(cmd)

    for data_set in data_sets:
        cmd = 'python analyze_results.py --experiment_path ' + experiment_path + ' --dataset ' + data_set + ' --cut_off ' + cut_off
        os.system(cmd)

    # dump everything into a single csv
    results_drive_train = osp.join('results', data_sets[0], experiment_path, 'perf', 'training_performance.csv')
    df_train = pd.read_csv(results_drive_train)
    df_train.drop('opt_t', axis=1, inplace=True)
    df_train.insert(loc=0, column='dataset', value=data_sets[0] + '_train')

    results_drive_val = osp.join('results', data_sets[0], experiment_path, 'perf', 'validation_performance.csv')
    df_val = pd.read_csv(results_drive_val)
    df_val.drop('opt_t', axis=1, inplace=True)
    df_val.insert(loc=0, column='dataset', value=data_sets[0] + '_val')

    perf_dfs = [df_train, df_val]
    for ds in data_sets:
        results = osp.join('results', ds, experiment_path, 'perf', 'test_performance.csv')
        df = pd.read_csv(results)
        df.insert(loc=0, column='dataset', value=ds)
        perf_dfs.append(df)
    n_out = osp.join('results', experiment_path.split('/')[1] + '_performance_summary.csv')
    df_all = pd.concat(perf_dfs).round(4).to_csv(n_out, index=None)
    