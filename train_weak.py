import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from itertools import chain

import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils import data
import gc

import torch.nn as nn

import wandb

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from src.data import DefaultDataset
from src.data import LoaderSampler

from src.resnet import ResNet_D
from src.unet import UNet

from src.tools import fig2img

from src.tools import freeze, unfreeze, weights_init_D, plot_Z_images, plot_random_Z_images

from src.tools import get_pushed_loader_stats, calculate_frechet_distance


T_ITERS = 10
f_LR, T_LR = 2e-4, 2e-4
IMG_SIZE = 64
BATCH_SIZE = 128
PLOT_INTERVAL = 100
COST = 'weak_mse'
CPKT_INTERVAL = 500
MAX_STEPS = 100001
SEED = 0x000000

ZC = 1
Z_STD = 0.1
Z_SIZE = 8
GAMMA0, GAMMA1 = 0.0, 0.66
GAMMA_ITERS = 500


mu_anime = np.load('/home/sudakovcom/Desktop/diffusion/NOT/stats/mu_anime.npy')
sigma_anime = np.load('/home/sudakovcom/Desktop/diffusion/NOT/stats/sigma_anime.npy')

assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:0')
torch.manual_seed(SEED)
np.random.seed(SEED)


transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_celeba = DefaultDataset('/home/sudakovcom/Desktop/diffusion/NOT/datasets/celeba_hq/train/female', transform=transform)
dataset_celeba_val = DefaultDataset('/home/sudakovcom/Desktop/diffusion/NOT/datasets/celeba_hq/val/female', transform=transform)
dataset_anime = DefaultDataset('/home/sudakovcom/Desktop/diffusion/NOT/datasets/anime_faces', transform=transform)

dataloader_celeba = data.DataLoader(dataset=dataset_celeba, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)
dataloader_celeba_val = data.DataLoader(dataset=dataset_celeba_val, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)
dataloader_anime = data.DataLoader(dataset=dataset_anime, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)

sampler_celeba = LoaderSampler(dataloader_celeba, device='cuda')
sampler_celeba_val = LoaderSampler(dataloader_celeba_val, device='cuda')
sampler_anime = LoaderSampler(dataloader_anime, device='cuda')

torch.cuda.empty_cache(); gc.collect()



f = ResNet_D(IMG_SIZE, nc=3).cuda()
f.apply(weights_init_D)

T = UNet(3+ZC, 3, base_factor=48).cuda() # ZC - noise input channels z

DEVICE_IDS = [0, 1]
if len(DEVICE_IDS) > 1:
    T = nn.DataParallel(T, device_ids=DEVICE_IDS)
    f = nn.DataParallel(f, device_ids=DEVICE_IDS)
    
T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)


X_fixed = sampler_celeba.sample(10)[:,None].repeat(1,4,1,1,1)
with torch.no_grad():
    Z_fixed = torch.randn(10, 4, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
    XZ_fixed = torch.cat([X_fixed, Z_fixed], dim=2)
del X_fixed, Z_fixed
Y_fixed = sampler_anime.sample(10)


# ! wandb login '05d5dbda72a88ed89fddab8ab96fe677f7590778'
run = wandb.init(name='weak_NOT', project='diffusion-NOT')


scaler = torch.GradScaler()

for step in tqdm(range(MAX_STEPS)):
    gamma = min(GAMMA1, GAMMA0 + (GAMMA1-GAMMA0) * step / GAMMA_ITERS)
    # T optimization
    unfreeze(T); freeze(f)
    for t_iter in range(T_ITERS): 
        T_opt.zero_grad()
        X = sampler_celeba.sample(BATCH_SIZE)[:,None].repeat(1,Z_SIZE,1,1,1)
        with torch.no_grad():
            Z = torch.randn(BATCH_SIZE, Z_SIZE, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        with torch.autocast(device_type='cuda', enabled=True):
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).permute(1,2,3,0).reshape(3, IMG_SIZE, IMG_SIZE, -1, Z_SIZE).permute(3,4,0,1,2)
            
            T_loss = F.mse_loss(X[:,0], T_XZ.mean(dim=1)).mean() - \
            f(T_XZ.flatten(start_dim=0, end_dim=1)).mean() + \
            T_XZ.var(dim=1).mean() * (1 - gamma - 1. / Z_SIZE)
            wandb.log({f'T_loss' : T_loss.item()}, step=step)
            scaler.scale(T_loss).backward()
            scaler.step(T_opt)
            scaler.update()
    del T_loss, T_XZ, X, Z; gc.collect(); torch.cuda.empty_cache()

    # f optimization
    freeze(T); unfreeze(f)
    X = sampler_celeba.sample(BATCH_SIZE)
    with torch.no_grad():
        Z = torch.randn(BATCH_SIZE, ZC, X.size(2), X.size(3), device='cuda') * Z_STD
        XZ = torch.cat([X,Z], dim=1)
        T_XZ = T(XZ)
    Y = sampler_anime.sample(BATCH_SIZE)
    f_opt.zero_grad()
    with torch.autocast(device_type='cuda', enabled=True):
        f_loss = f(T_XZ).mean() - f(Y).mean()
        scaler.scale(f_loss).backward()
        scaler.step(f_opt)
        scaler.update()
        wandb.log({f'f_loss' : f_loss.item()}, step=step)
    del f_loss, Y, X, T_XZ, Z, XZ; gc.collect(); torch.cuda.empty_cache()
        
    if step % PLOT_INTERVAL == 0:
        clear_output(wait=True)
        print(f'step {step} of {MAX_STEPS}')
        print('Plotting')
        
        fig, axes = plot_Z_images(XZ_fixed, Y_fixed, T)
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
        
        fig, axes = plot_random_Z_images(sampler_celeba, ZC, Z_STD, sampler_anime, T)
        wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        
        

    if step % CPKT_INTERVAL == 0:        
        torch.save(T.state_dict(), f'T_{step}.pt')
        torch.save(f.state_dict(), f'f_{step}.pt')
        torch.save(f_opt.state_dict(), f'f_opt_{step}.pt')
        torch.save(T_opt.state_dict(), f'T_opt_{step}.pt')

        artifact = wandb.Artifact('T', type='model')
        artifact.add_file(f'T_{step}.pt')
        run.log_artifact(artifact)
        artifact = wandb.Artifact('f', type='model')
        artifact.add_file(f'f_{step}.pt')
        run.log_artifact(artifact)
        artifact = wandb.Artifact('T_opt', type='model')
        artifact.add_file(f'T_opt_{step}.pt')
        run.log_artifact(artifact)
        artifact = wandb.Artifact('f_opt', type='model')
        artifact.add_file(f'f_opt_{step}.pt')
        run.log_artifact(artifact)
    
    gc.collect(); torch.cuda.empty_cache()


