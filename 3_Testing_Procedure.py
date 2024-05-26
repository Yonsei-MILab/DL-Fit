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
import math
from torch import nn
from 2_Util_Phase_Only_Joint import laplacian_fn, mag_weight_fn, neural_weight_fn, build_net, train_net, Uncertainty_mask

# Configure GPU
configure(gpu_id=0)
device = torch.device("cuda")

# Testing Network
def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

def build_net_test(channel_size, num_layers, kernel_size):
    in_size = out_size = prod(kernel_size)
    sizes = (in_size,) + (channel_size,) * (num_layers - 1) + (out_size,)
    layers = []
    for i, (insz, outsz) in enumerate(zip(sizes, sizes[1:])):
        layers += [nn.Linear(insz, outsz)]
        if i != num_layers - 1:
            layers += [nn.ReLU(inplace=True)]        
            layers += [nn.Dropout(0.4)]
        else:
            layers += [nn.Sigmoid()]
    
    return nn.Sequential(*layers)

wf = 2 * np.pi * 3 * 42.576 * (10 ** 6)
mu = 4 * np.pi * (10 ** (-7))
muwf = mu * wf
res = 0.002, 0.002
kernel_size = 17, 17

channel_size = 2048
num_layers = 4
learning_rate = 1e-4

net1 = build_net_test(channel_size, num_layers, kernel_size).to(device)
net2 = build_net_test(channel_size, num_layers, kernel_size).to(device)
net3 = build_net_test(channel_size, num_layers, kernel_size).to(device)

net1 = net1.eval()
net2 = net2.eval()
net3 = net3.eval()

net_path1 = "DL_Fit_Axial_Plane_Coupling.pt"
net1.load_state_dict(torch.load(net_path1))

net_path2 = "DL_Fit_Coronal_Plane_Coupling.pt"
net2.load_state_dict(torch.load(net_path2))

net_path3 = "DL_Fit_Sagittal_Plane_Coupling.pt"
net3.load_state_dict(torch.load(net_path3))

# Constants
wf = 2 * np.pi * 3 * 42.576 * (10 ** 6)
mu = 4 * np.pi * (10 ** (-7))
muwf = mu * wf
res = 0.002, 0.002
kernel_size = 17, 17

# Load Test Dataset
data_test = loadmat("test_dataset.mat", simplify_cells=True)
imgs_test, masks_test = [
    data_test[k] for k in ["dataset_training", "dataset_mask"]
]
imgs_test, masks_test = (
    torch.tensor(imgs_test, dtype=torch.complex64),
    torch.tensor(masks_test, dtype=bool),
)
imgs_test, masks_test = [x.to(device) for x in [imgs_test, masks_test]]

phase_test = torch.angle(imgs_test)
mag_test = torch.abs(imgs_test)

# Test for Axial Plane
DL_result_Axial = []
for idx in range(imgs3.size(0)):
    imgs_idx = imgs_test[idx]
    mask_idx = mag_test[idx]
    phase_idx = torch.angle(imgs_idx)
    mag_idx = torch.abs(imgs_idx)
    
    lap_fn = laplacian_fn(mask_idx, kernel_size, res)
    mean_idx, std_idx = torch.mean(mag_idx), torch.std(mag_idx)
    
    with torch.no_grad():
        cond_nn = lap_fn(phase_idx /2, neural_weight_fn(mag_idx, mask_idx, net1, mean_idx, std_idx)) / muwf
        DL_result = torch.where(cond_nn > 0, cond_nn2, lap_fn(phase_idx / 2, mag_weight_fn(mag_idx, mask_idx, sigma=0.2)) / muwf)
        DL_result_Axial.append(DL_result)
DL_result_Axial = torch.stack(DL_result_Axial) 





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
net_path = "DL_Fit_xy_plane.pt"
torch.save(net.state_dict(), net_path)
