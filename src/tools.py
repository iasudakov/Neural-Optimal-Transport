import pandas as pd
import numpy as np
import random
import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm_notebook as tqdm
from .fid_score import calculate_frechet_distance
from .distributions import LoaderSampler
import torchvision.datasets as datasets
import torchvision
import h5py
from torch.utils.data import TensorDataset, ConcatDataset

import gc

from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST


class ColoredMNIST(MNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hues = 360 * torch.rand(super().__len__())

    def __len__(self):
        return super().__len__()

    def color_image(self, img, idx):
        img_min = 0
        a = (img - img_min) * (self.hues[idx] % 60) / 60
        img_inc = a
        img_dec = img - a

        colored_image = torch.zeros((3, img.shape[1], img.shape[2]))
        H_i = round(self.hues[idx].item() / 60) % 6

        if H_i == 0:
            colored_image[0] = img
            colored_image[1] = img_inc
            colored_image[2] = img_min
        elif H_i == 1:
            colored_image[0] = img_dec
            colored_image[1] = img
            colored_image[2] = img_min
        elif H_i == 2:
            colored_image[0] = img_min
            colored_image[1] = img
            colored_image[2] = img_inc
        elif H_i == 3:
            colored_image[0] = img_min
            colored_image[1] = img_dec
            colored_image[2] = img
        elif H_i == 4:
            colored_image[0] = img_inc
            colored_image[1] = img_min
            colored_image[2] = img
        elif H_i == 5:
            colored_image[0] = img
            colored_image[1] = img_min
            colored_image[2] = img_dec

        return colored_image

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return self.color_image(img, idx), label


def get_random_colored_images(images, seed = 0x000000):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
    colored_images = 2*colored_images - 1
    
    return colored_images
    

def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')    

    return TensorDataset(dataset, torch.zeros(len(dataset)))



class ZeroImageDataset(Dataset):
    def __init__(self, n_channels, h, w, n_samples, transform=None):
        self.n_channels = n_channels
        self.h = h
        self.w = w
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.ones(self.n_channels, self.h, self.w), torch.zeros(self.n_channels, self.h, self.w)
    
    
    
from pathlib import Path
from itertools import chain
from PIL import Image
from torch.utils import data

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
    
    
    
class TenDataset(data.Dataset):
    def __init__(self, X, transform=None):
        self.samples = X

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.samples.shape[0]
    
    


def load_dataset(name, img_size=64, batch_size=64, 
                 shuffle=True, device='cuda', return_dataset=False,
                 num_workers=0):
    if name.startswith('anime_faces'):
        
        transform = Compose([
            Resize((img_size, img_size)), 
            ToTensor(), 
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = DefaultDataset(f'./data/anime_faces', transform=transform)
        
        train_ratio = 0.9
        test_ratio = 0.1
        
        train_size = int(len(dataset) * train_ratio)
        test_size = int(len(dataset) * test_ratio)
        idx = np.arange(len(dataset))
        
        np.random.seed(0x000000); np.random.shuffle(idx)
        
        train_idx = idx[:train_size]
        test_idx = idx[-test_size:]
        
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)  
        
    elif name.startswith('celeba_hq'):
        
        transform = Compose([
            Resize((img_size, img_size)), 
            ToTensor(), 
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = DefaultDataset(f'./data/celeba_hq/train/female', transform=transform)
        test_set = DefaultDataset(f'./data/celeba_hq/val/female', transform=transform)
        
    elif name.startswith('male'):
        
        transform = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = DefaultDataset(f'./data/male{img_size}train', transform=transform)
        test_set = DefaultDataset(f'./data/male{img_size}test', transform=transform)
        
    elif name.startswith('female'):
        
        transform = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = DefaultDataset(f'./data/female{img_size}train', transform=transform)
        test_set = DefaultDataset(f'./data/female{img_size}test', transform=transform)
        
    elif name.startswith("MNIST"):
        # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
        transform = torchvision.transforms.Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            torchvision.transforms.Lambda(lambda x: 2 * x - 1)
        ])
        
        dataset_name = name.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"
        
        classes = [int(number) for number in name.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]

        train_set = datasets.MNIST(f'./data/{name}', train=True, transform=transform, download=True)
        test_set = datasets.MNIST(f'./data/{name}', train=False, transform=transform, download=True)
        
        train_test = []
        
        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(torch.stack(
                    [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                    dim=0
                ))
                labels += [k]*data[-1].shape[0]
            data = torch.cat(data, dim=0)
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)
            
            if is_colored:
                data = get_random_colored_images(data)
            
            train_test.append(TenDataset(data))
            
        train_set, test_set = train_test  
    else:
        raise Exception('Unknown dataset')
    
    if return_dataset:
        return train_set, test_set
        
    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size), device)
    return train_sampler, test_sampler


def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')    

    return TensorDataset(dataset, torch.zeros(len(dataset)))


def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                                           
                    batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()

                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                        
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


    
    with torch.no_grad():
        for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                TX = T(X[start:end].type(torch.FloatTensor).to(device))
                l2_sum += ((X[start:end].type(torch.FloatTensor).to(device) - TX)**2).sum()
                lpips_sum += lpips(normalize(TX).cpu(), X[start:end].type(torch.FloatTensor))
                batch = TX.add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
                val_size += end - start

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma, l2_sum/val_size, lpips_sum/val_size

def get_Z_pushed_loader_stats(T, loader, ZC=1, Z_STD=0.1, batch_size=8, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction = 'sum')
    
    l2_sum = 0
    lpips_sum = 0
    val_size = 0
    
    with torch.no_grad():
        for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            Z = torch.randn(len(X), ZC, X.size(2), X.size(3)) * Z_STD
            XZ = torch.cat([X,Z], dim=1)
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                TX = T(XZ[start:end].type(torch.FloatTensor).to(device))
                l2_sum += ((X[start:end].type(torch.FloatTensor).to(device) - TX)**2).sum()
                lpips_sum += lpips(normalize(TX).cpu(), X[start:end].type(torch.FloatTensor))
                batch = TX.add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
                val_size += end - start

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma, l2_sum/val_size, lpips_sum/val_size


def get_enot_sde_pushed_loader_stats(T, loader, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model);
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(
                        X[start:end].type(torch.FloatTensor).to(device)
                    )[0].add(1).mul(.5)
                    
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_sde_pushed_loader_stats(T, loader, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(
                        X[start:end].type(torch.FloatTensor).to(device)
                    )[0][:, -1].add(1).mul(.5)
                    
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma




import matplotlib.pyplot as plt
from scipy import linalg
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



def plot_images(X, Y, T):
    freeze(T);
    with torch.no_grad():
        T_X = T(X)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_images_EDM(X, Y, T):
    freeze(T);
    with torch.no_grad():
        latent_z = torch.randn(10, 100, device=X.device)*0.1
        T_X = T(X, latent_z)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)

def plot_random_images_EDM(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images_EDM(X, Y, T)



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
    

def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]
    
    
def get_pushed_loader_stats(T, loader, batch_size=8, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction = 'sum')
    
    pred_arr = []
    
    l2_sum = 0
    lpips_sum = 0
    val_size = 0
    
    with torch.no_grad():
        for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                TX = T(X[start:end].type(torch.FloatTensor).to(device))
                l2_sum += ((X[start:end].type(torch.FloatTensor).to(device) - TX)**2).sum()
                lpips_sum += lpips(normalize(TX).cpu(), X[start:end].type(torch.FloatTensor))
                batch = TX.add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
                val_size += end - start

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma, l2_sum/val_size, lpips_sum/val_size


def get_pushed_loader_stats_EDM(T, loader, batch_size=8, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction = 'sum')
    
    pred_arr = []
    
    l2_sum = 0
    lpips_sum = 0
    val_size = 0
    
    with torch.no_grad():
        for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                latent_z = torch.randn(end-start, 100, device=device)*0.1
                TX = T(X[start:end].type(torch.FloatTensor).to(device), latent_z)
                l2_sum += ((X[start:end].type(torch.FloatTensor).to(device) - TX)**2).sum()
                lpips_sum += lpips(normalize(TX).cpu(), X[start:end].type(torch.FloatTensor))
                batch = TX.add(1).mul(.5)
                pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
                val_size += end - start

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma, l2_sum/val_size, lpips_sum/val_size

def plot_Z_images(XZ, Y, T):
    freeze(T);
    with torch.no_grad():
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1,2,3,0).reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4).permute(4,3,0,1,2).flatten(start_dim=0, end_dim=1)
        imgs = torch.cat([XZ[:,0,:Y.shape[1]], T_XZ, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(4):
        axes[i+1, 0].set_ylabel('T(X,Z)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_Z_images(X_sampler, ZC, Z_STD, Y_sampler, T):
    X = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_Z_images(XZ, Y, T)
