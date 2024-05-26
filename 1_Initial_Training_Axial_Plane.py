# Import necessary libraries
%load_ext autoreload
%autoreload 2

import copy
from itertools import cycle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
from skimage.metrics import normalized_root_mse as nrmse, structural_similarity as ssim, peak_signal_noise_ratio as psnr
from util import configure
from 1_Util_Phase_Only import laplacian_fn, mag_weight_fn, neural_weight_fn, build_net, train_net

# Configure GPU
configure(gpu_id=0)
device = torch.device("cuda")

# Load training dataset
data = loadmat("training_dataset.mat", simplify_cells=True)
gts, masks, imgs = [data[k] for k in ["dataset_label", "dataset_mask", "dataset_training"]]

# Convert to torch tensors and move to device
gts, masks, imgs = (
    torch.tensor(gts, dtype=torch.float),
    torch.tensor(masks, dtype=bool),
    torch.tensor(imgs, dtype=torch.complex64),
)
gts, masks, imgs = [x.to(device) for x in [gts, masks, imgs]]

# Constants
wf = 2 * np.pi * 3 * 42.576 * (10 ** 6)
mu = 4 * np.pi * (10 ** (-7))
muwf = mu * wf
res = 0.002, 0.002
kernel_size = 17, 17

# Reshape data
new_shape = (gts.size(0) * gts.size(1), gts.size(2), gts.size(3))
gts = gts.reshape(new_shape)
masks = masks.reshape(new_shape)
imgs = imgs.reshape(new_shape)

# Data Augmentation
def augment_data(tensor):
    flipped_slices_lr = torch.flip(tensor, dims=[2])
    flipped_slices_ud = torch.flip(tensor, dims=[1])
    flipped_slices_lr_ud = torch.flip(tensor, dims=[2])
    return torch.cat([tensor, flipped_slices_lr, flipped_slices_ud, flipped_slices_lr_ud], dim=0)

imgs = augment_data(imgs)
masks = augment_data(masks)
gts = augment_data(gts)

# Prepare dataset
phases, mags = torch.angle(imgs), torch.abs(imgs)
mean, std = torch.mean(mags), torch.std(mags)
dataset = cycle(zip(imgs, masks, gts))

# Iteration hook to track losses
class IterHook:
    def __init__(self):
        self.losses = []

    def __call__(self, i, loss):
        self.losses.append({"Iteration": i, "Loss": loss})

# Training parameters
channel_size = 2048
num_layers = 4
num_iters = 48000
learning_rate = 1e-4
iter_hook = IterHook()

# Build and train network
net = build_net(channel_size, num_layers, kernel_size).to(device)
net = train_net(net, dataset, num_iters, learning_rate, kernel_size, res, muwf, iter_hook)

# Save losses and model
losses = pd.DataFrame(iter_hook.losses)
net_path = "DL_Fit_Axial_Plane.pt"
torch.save(net.state_dict(), net_path)
