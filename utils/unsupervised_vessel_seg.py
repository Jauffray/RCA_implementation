import numpy as np
from skimage.restoration import denoise_nl_means
from skimage.transform import rotate
from skimage.morphology import black_tophat

def bottom_hat_rotated(im, length, angle):
    base_struct = np.zeros((length+1, length+1))
    base_struct[length//2, :] = 1
    new_struct = rotate(base_struct, angle, order=0).astype(bool)
    return black_tophat(im, new_struct)

def find_vessels_unsupervised(image):
    # set up values for strucutring element construction
    angles = np.linspace(0, 180, num=12, endpoint=False)
    length = 16
    # extract green channel
    im_green = np.array(image)[:,:,1]
    # denoise it
    denoised = denoise_nl_means(im_green, patch_size=3, patch_distance=21, h=3, multichannel=False)
    # build segmentation
    result = np.zeros(im_green.shape)
    for angle in angles:
        result += bottom_hat_rotated(denoised, length, angle)
    # map values to [0,1] and return
    result = (result - result.min())/ (result.max() - result.min())
    return result
