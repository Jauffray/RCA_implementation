import os

cfg_files = os.listdir('results/')
cfg_files = [cfg for cfg in cfg_files if '.cfg' in cfg]

for cfg_file in cfg_files:
    perf_summ_name = 'results/'+cfg_file[:-11]+'_performance_summary.csv'
    if os.path.isfile(perf_summ_name): continue
    base_cmd = 'python bunch_evaluation.py --config_file results/'+ cfg_file
    os.system(base_cmd)

