import torch
import sys, json, os, argparse

from PIL import Image
from matplotlib import pyplot as plt

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='False', help='image name')

if __name__ == '__main__':

    args = parser.parse_args()
    # reproducibility
    import numpy as np
    # gather parser parameters
    img = args.img

    if(img != 'False'):
        img_array = np.load('results/STARE/experiments/2020-01-22-01-1647/'+img)

        plt.imshow(img_array, cmap='gray')
        plt.show()
        # im = Image.fromarray(img_array)
        # im.show()
