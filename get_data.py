import os
import os.path as osp
import shutil
import pandas as pd
from PIL import Image
import numpy as np

print('downloading data')
call = '(mkdir data && cd data && curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data)'
os.system(call)

# process drive data, generate CSVs
path_ims = 'data/DRIVE/images'
path_masks = 'data/DRIVE/mask'
path_gts = 'data/DRIVE/manual'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))

# append paths
num_ims = len(all_im_names)
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

test_im_names = all_im_names[:num_ims//2]
train_im_names = all_im_names[num_ims//2:]

test_mask_names = all_mask_names[:num_ims//2]
train_mask_names = all_mask_names[num_ims//2:]

test_gt_names = all_gt_names[:num_ims//2]
train_gt_names = all_gt_names[num_ims//2:]


df_drive_train = pd.DataFrame({'im_paths': train_im_names,
                               'gt_paths': train_gt_names,
                               'mask_paths': train_mask_names})

df_drive_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                               'mask_paths': test_mask_names})

df_drive_train, df_drive_val = df_drive_train[:16], df_drive_train[16:]


df_drive_train.to_csv('data/DRIVE/train.csv', index=False)
df_drive_val.to_csv('data/DRIVE/val.csv', index=False)
df_drive_test.to_csv('data/DRIVE/test.csv', index=False)
print('DRIVE prepared')

src = 'missing_masks/stare-masks/'
dst = 'data/STARE/stare-masks/'
# os.makedirs(dst, exist_ok=True)
shutil.copytree(src, dst) #copytree creates the dir


path_ims = 'data/STARE/stare-images'
path_masks = 'data/STARE/stare-masks'
path_gts = 'data/STARE/labels-ah'

test_im_names = sorted(os.listdir(path_ims))
test_mask_names = sorted(os.listdir(path_masks))
test_gt_names = sorted(os.listdir(path_gts))

# append paths
num_ims = len(all_im_names)
test_im_names = [osp.join(path_ims, n) for n in test_im_names]
test_mask_names = [osp.join(path_masks, n) for n in test_mask_names]
test_gt_names = [osp.join(path_gts, n) for n in test_gt_names]

df_stare_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                              'mask_paths': test_mask_names})
df_stare_test.to_csv('data/STARE/test.csv', index=False)
print('STARE prepared')

path_ims = 'data/AV-WIDE/images'
path_masks = 'data/AV-WIDE/masks'
os.makedirs(path_masks, exist_ok=True)
path_gts = 'data/AV-WIDE/manual'

test_im_names = sorted(os.listdir(path_ims))
test_gt_names = sorted(os.listdir(path_gts))

for n in test_im_names:
    im = Image.open(osp.join(path_ims, n))
    mask = 255*np.ones((im.size[1], im.size[0]), dtype=np.uint8)
    h, w = mask.shape
    h_margin, w_margin = int(0.01 * h), int(0.01 * w)
    mask[: h_margin, :] = 0
    mask[:, : w_margin] = 0

    mask[h - h_margin:, :] = 0
    mask[:, w - w_margin:] = 0
    Image.fromarray(mask).save(osp.join(path_masks, n))

num_ims = len(all_im_names)
test_mask_names = [osp.join(path_masks, n) for n in test_im_names]
test_im_names = [osp.join(path_ims, n) for n in test_im_names]
test_gt_names = [osp.join(path_gts, n) for n in test_gt_names]

df_wide_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                              'mask_paths': test_mask_names})

df_wide_test.to_csv('data/AV-WIDE/test.csv', index=False)
print('AV-WIDE prepared')


src = 'missing_masks/chase-masks/'
dst = 'data/CHASEDB/chase-masks/'
# os.makedirs(dst, exist_ok=True)
shutil.copytree(src, dst) #copytree creates the dir

path_ims = 'data/CHASEDB/images'
path_masks = 'data/CHASEDB/chase-masks'
path_gts = 'data/CHASEDB/manual'

test_im_names = sorted(os.listdir(path_ims))
test_mask_names = sorted(os.listdir(path_masks))
test_gt_names = sorted(os.listdir(path_gts))

# append paths
num_ims = len(all_im_names)
test_im_names = [osp.join(path_ims, n) for n in test_im_names]
test_mask_names = [osp.join(path_masks, n) for n in test_mask_names]
test_gt_names = [osp.join(path_gts, n) for n in test_gt_names if '1st' in n]

df_chase_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                              'mask_paths': test_mask_names})

df_chase_test.to_csv('data/CHASEDB/test.csv', index=False)
print('CHASE-DB prepared')

os.makedirs('experiments', exist_ok=True)
os.makedirs('results', exist_ok=True)

# remove junk
shutil.rmtree('data/VEVIO')
shutil.rmtree('data/DRIVE/splits')
shutil.rmtree('data/STARE/splits')
shutil.rmtree('data/AV-WIDE/splits')
shutil.rmtree('data/CHASEDB/splits')
# shutil.rmtree('missing_masks')
