import matplotlib.pyplot as plt
# %matplotlib inline 
import numpy as np
import torch

import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils import data
import gc

import wandb

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output


from src.data import DefaultDataset
from src.data import LoaderSampler

from src.cond_resnet import ResNet_D
from src.unet import UNet

from src.tools import fig2img

from src.tools import freeze, unfreeze, weights_init_D, plot_images, plot_random_images

from src.tools import get_pushed_loader_stats, calculate_frechet_distance

T_ITERS = 10
f_LR, T_LR = 2e-4, 2e-4
IMG_SIZE = 128
BATCH_SIZE = 128
PLOT_INTERVAL = 50
COST = 'mse' # Mean Squared Error
CPKT_INTERVAL = 1000
MAX_STEPS = 10000
SEED = 0x000000
sigma_min=0.02
sigma_max=100


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

T = UNet(3, 3, base_factor=48).cuda()
    
T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)


for i in range(7): #подобрал чтобы лица смотрели прямо
    X_fixed = sampler_celeba.sample(10)
    Y_fixed = sampler_anime.sample(10)


wandb.init(name='noised_strong_NOT', project='diffusion-NOT')



scaler = torch.GradScaler()

for step in tqdm(range(MAX_STEPS)):
    print('step:', step)
    # T optimization
    unfreeze(T); freeze(f)
    for t_iter in range(T_ITERS):
        T_opt.zero_grad()
        X = sampler_celeba.sample(BATCH_SIZE)
        with torch.autocast(device_type='cuda', enabled=True):
            T_X = T(X)
            if COST == 'mse':
                rnd_normal = torch.randn(X.shape[0], device=X.device)
                sigma = torch.exp(rnd_normal)[:, None, None, None] # [batch, 1, 1, 1]
                weight = (sigma ** 2 + 0.5 ** 2) / (sigma ** 2 * 0.5 ** 2) # [batch, 1, 1, 1]
                n = torch.randn_like(X) * sigma
                T_loss = ((F.mse_loss(X, T_X) - f(T_X + n, sigma.flatten())) * weight).mean()
            else:
                raise Exception('Unknown COST')
            scaler.scale(T_loss).backward()
            scaler.step(T_opt)
            scaler.update()
    wandb.log({f'T_loss' : T_loss.item()}, step=step) 
    del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()

    # f optimization
    freeze(T); unfreeze(f)
    X = sampler_celeba.sample(BATCH_SIZE)
    with torch.no_grad():
        T_X = T(X)
    Y = sampler_anime.sample(BATCH_SIZE)
    f_opt.zero_grad()
    with torch.autocast(device_type='cuda', enabled=True):
        rnd_normal = torch.randn(X.shape[0], device=X.device)
        sigma = torch.exp(rnd_normal)[:, None, None, None] # [batch, 1, 1, 1]
        weight = (sigma ** 2 + 0.5 ** 2) / (sigma ** 2 * 0.5 ** 2) # [batch, 1, 1, 1]
        n = torch.randn_like(X) * sigma
        f_loss = ((f(T_X + n, sigma.flatten()) - f(Y + n, sigma.flatten())) * weight).mean()
        
        scaler.scale(f_loss).backward()
        scaler.step(f_opt)
        scaler.update()
    wandb.log({f'f_loss' : f_loss.item()}, step=step) 
    del f_loss, Y, X, T_X; gc.collect(); torch.cuda.empty_cache()
    
    clear_output(wait=True)
        
    if step % PLOT_INTERVAL == 0:
        clear_output(wait=True)
        print(f'step {step} of {MAX_STEPS}')
        print('Plotting')
        
        fig, axes = plot_images(X_fixed, Y_fixed, T)
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
        
        fig, axes = plot_random_images(sampler_celeba, sampler_anime, T)
        wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step) 
        
        mu, sigma, l2, lpips = get_pushed_loader_stats(T, sampler_celeba_val.loader)
        fid = calculate_frechet_distance(mu_anime, sigma_anime, mu, sigma)
        wandb.log({f'FID' : fid}, step=step)
        wandb.log({f'L2' : l2}, step=step)
        wandb.log({f'LPIPS' : lpips}, step=step)
        del mu, sigma, fid, lpips

    if step % CPKT_INTERVAL == 0:
        
        torch.save(T.state_dict(), f'/home/sudakovcom/Desktop/diffusion/NOT/checkpoints/T_noised_{step}.pt')
        torch.save(f.state_dict(), f'/home/sudakovcom/Desktop/diffusion/NOT/checkpoints/f_noised_{step}.pt')
        torch.save(f_opt.state_dict(), f'/home/sudakovcom/Desktop/diffusion/NOT/checkpoints/f_noised_opt_{step}.pt')
        torch.save(T_opt.state_dict(), f'/home/sudakovcom/Desktop/diffusion/NOT/checkpoints/T_noised_opt_{step}.pt')
    
    gc.collect(); torch.cuda.empty_cache()
