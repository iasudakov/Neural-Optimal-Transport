import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import wandb

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from src.resnet import ResNet_D
# from EDM_src.discriminator import Discriminator_small
from EDM_src.SongUnet import SongUNet

from src.tools import fig2img
from src.tools import weights_init_D, plot_images_EDM, plot_random_images_EDM
from src.tools import get_pushed_loader_stats_EDM
from src.tools import unfreeze, freeze

from src.tools import load_dataset
from src.fid_score import calculate_frechet_distance
from src.tools import weights_init_D
import json

import warnings
warnings.filterwarnings('ignore')



T_ITERS = 10
NZ = 100
f_LR, T_LR = 1e-4, 1e-4
IMG_SIZE = 32
BATCH_SIZE = 64
PLOT_INTERVAL = 50
COST = 'mse'
CPKT_INTERVAL = 1000
MAX_STEPS = 10000
SEED = 0x000000


assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:0')
torch.manual_seed(SEED)
np.random.seed(SEED)

# DATASET1  = 'celeba_hq'
# DATASET2 = 'anime_faces'
DATASET1 = 'MNIST-colored_2'
DATASET2 = 'MNIST-colored_3'


filename = './stats/{}_{}_test.json'.format(DATASET2, IMG_SIZE)
with open(filename, 'r') as fp:
    data_stats = json.load(fp)
    mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
del data_stats

X_sampler, X_test_sampler = load_dataset(DATASET1, img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=8)
Y_sampler, Y_test_sampler = load_dataset(DATASET2, img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=8)
    
torch.cuda.empty_cache(); gc.collect()

f = ResNet_D(IMG_SIZE, 3).cuda()
f.apply(weights_init_D)

T = SongUNet(IMG_SIZE, 3, 3).cuda()

DEVICE_IDS = [0]
if len(DEVICE_IDS) > 1:
    T = nn.DataParallel(T, device_ids=DEVICE_IDS)
    f = nn.DataParallel(f, device_ids=DEVICE_IDS)
    
T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)

for i in range(7): #подобрал чтобы лица смотрели прямо
    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    
wandb.init(name='strong_NOT_tester', project='diffusion-NOT')


scaler = torch.GradScaler()

for step in tqdm(range(MAX_STEPS)):
    # T optimization
    unfreeze(T); freeze(f)
    for t_iter in range(T_ITERS):
        T_opt.zero_grad()
        X = X_sampler.sample(BATCH_SIZE)
        with torch.autocast(device_type='cuda', enabled=True):
            latent_z = torch.randn(BATCH_SIZE, NZ, device=X.device)*0.1
            T_X = T(X, latent_z)
            T_loss = F.mse_loss(X, T_X).mean() - f(T_X).mean()
            scaler.scale(T_loss).backward()
            scaler.step(T_opt)
            scaler.update()
    wandb.log({f'T_loss' : T_loss.item()}, step=step) 
    del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()

    # f optimization
    freeze(T); unfreeze(f)
    X = X_sampler.sample(BATCH_SIZE)
    with torch.no_grad():
        latent_z = torch.randn(BATCH_SIZE, NZ, device=X.device)*0.1
        T_X = T(X, latent_z)
    Y = Y_sampler.sample(BATCH_SIZE)
    f_opt.zero_grad()
    with torch.autocast(device_type='cuda', enabled=True):
        f_loss = f(T_X).mean() - f(Y).mean()
        scaler.scale(f_loss).backward()
        scaler.step(f_opt)
        scaler.update()
    wandb.log({f'f_loss' : f_loss.item()}, step=step) 
    del f_loss, Y, X, T_X; gc.collect(); torch.cuda.empty_cache()
        
    if step % PLOT_INTERVAL == 0:
        clear_output(wait=True)
        print(f'step {step} of {MAX_STEPS}')
        print('Plotting')
        
        fig, axes = plot_images_EDM(X_fixed, Y_fixed, T)
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
        
        fig, axes = plot_random_images_EDM(X_sampler, Y_sampler, T)
        wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
        
        mu, sigma, l2, lpips = get_pushed_loader_stats_EDM(T, X_test_sampler.loader)
        fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
        wandb.log({f'FID' : fid}, step=step)
        wandb.log({f'L2' : l2}, step=step)
        wandb.log({f'LPIPS' : lpips}, step=step)
        del mu, sigma, fid, lpips

    if step % CPKT_INTERVAL == 0:
        torch.save(T.state_dict(), f'./checkpoints/EDM_strong_NOT_D/T_{step}.pt')
        torch.save(f.state_dict(), f'./checkpoints/EDM_strong_NOT_D/f_{step}.pt')
        torch.save(f_opt.state_dict(), f'./checkpoints/EDM_strong_NOT_D/f_opt_{step}.pt')
        torch.save(T_opt.state_dict(), f'./checkpoints/EDM_strong_NOT_D/T_opt_{step}.pt')
    
    gc.collect(); torch.cuda.empty_cache()
    
    