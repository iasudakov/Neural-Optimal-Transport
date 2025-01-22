import os, sys
sys.path.append(".")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src.tools import unfreeze, freeze
from src.tools import load_dataset, get_loader_stats

from copy import deepcopy
import json

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


gc.collect(); torch.cuda.empty_cache()

DEVICE_ID = 0

DATASET_LIST = [
    # ('celeba_hq', 64, 1),
    # ('anime_faces', 64, 1),
    ('MNIST-colored_3', 32, 1),
]

assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')


for DATASET, IMG_SIZE, N_EPOCHS in tqdm(DATASET_LIST):
    print('Processing {}'.format(DATASET))
    sampler, _ = load_dataset(DATASET, img_size=IMG_SIZE, batch_size=256)
    print('Dataset {} loaded'.format(DATASET))

    mu, sigma = get_loader_stats(sampler.loader, n_epochs=N_EPOCHS, verbose=True, batch_size=256)
    print('Trace of sigma: {}'.format(np.trace(sigma)))
    stats = {'mu' : np.array(mu), 'sigma' : np.array(sigma)}
    print('Stats computed')

    filename = './stats/{}{}train'.format(DATASET, IMG_SIZE)
    np.savez(filename, **stats)
    print('States saved to {}'.format(filename))