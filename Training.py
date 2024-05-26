%load_ext autoreload
%autoreload 2
import copy
from itertools import cycle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter, gaussian_filter
from datetime import datetime
from dataclasses import dataclass
import random
from typing import Optional
#from torchmetrics import StructuralSimilarityIndexMeasure
import os
import sys
import matplotlib as mpl
import matplotlib_inline as mpli
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Tuple
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from skimage.transform import resize as skresize
from skimage.metrics import (
    normalized_root_mse as nrmse,
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)

# specify which GPU(s) to be used
from util import configure
configure(gpu_id=0)
import torch
from Implementation_sigmoid import laplacian_fn, mag_weight_fn, neural_weight_fn, build_net, train_net, Uncertainty_mask
device = torch.device("cuda")

# Axial Plane
## Load training dataset
data = loadmat("training_dataset.mat", simplify_cells=True)

## dataset_label: Simulated ground-truth conductivity data, dataset_mask: Brain mask data, dataset_training: Simulated complex-valued MRI data
gts, masks, imgs = [
    data[k] for k in ["dataset_label", "dataset_mask", "dataset_training"]
]
gts, masks, imgs = (
    torch.tensor(gts, dtype=torch.float),
    torch.tensor(masks, dtype=bool),
    torch.tensor(imgs, dtype=torch.complex64),
)
gts, masks, imgs = [x.to(device) for x in [gts, masks, imgs]]
wf = 2 * np.pi * 3 * 42.576 * (10 ** 6)
mu = 4 * np.pi * (10 ** (-7))
muwf = mu * wf
res = 0.002, 0.002
kernel_size = 17, 17

gts = gts.reshape(-1, gts.size(-2), gts.size(-1))
masks = gts.reshape(-1, gts.size(-2), gts.size(-1))
imgs = gts.reshape(-1, gts.size(-2), gts.size(-1))

## Data Augmentation
flipped_slices_lr = torch.flip(imgs, dims=[2])  
flipped_slices_ud = torch.flip(imgs, dims=[1])  
flipped_slices_lr_ud = torch.flip(imgs, dims=[2])  
imgs = torch.cat([imgs, flipped_slices_lr, flipped_slices_ud, flipped_slices_lr_ud], dim=0)
masksflipped_slices_lr = torch.flip(masks, dims=[2])  
masksflipped_slices_ud = torch.flip(masks, dims=[1])  
masksflipped_slices_lr_ud = torch.flip(masks, dims=[2])  
masks = torch.cat([masks, masksflipped_slices_lr, masksflipped_slices_ud, masksflipped_slices_lr_ud], dim=0)
gtsflipped_slices_lr = torch.flip(gts, dims=[2]) 
gtsflipped_slices_ud = torch.flip(gts, dims=[1]) 
gtsflipped_slices_lr_ud = torch.flip(gts, dims=[2]) 
gts = torch.cat([gts, gtsflipped_slices_lr, gtsflipped_slices_ud, gtsflipped_slices_lr_ud], dim=0)

phases, mags = torch.angle(imgs), torch.abs(imgs)
mean, std = torch.mean(mags), torch.std(mags)
dataset = cycle(zip(imgs, masks, gts))

class IterHook:
    def __init__(self):
        self.losses = []

    def __call__(self, i, loss):
        self.losses += [{"Iteration": i, "Loss": loss}]

channel_size = 2048
num_layers = 4
num_iters = 48000
learning_rate = 1e-4
iter_hook = IterHook()
net = build_net(channel_size, num_layers, kernel_size).to(device)
net = train_net(
    net, dataset, num_iters, learning_rate, kernel_size, res, muwf, iter_hook
)
losses = pd.DataFrame(iter_hook.losses)

net_path = "DL_xy_sigmoid_non_flip4.pt"
torch.save(net.state_dict(), net_path)
