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
from 2_Util_Phase_Only_Joint import laplacian_fn, mag_weight_fn, neural_weight_fn, build_net, train_net, Uncertainty_mask
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

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
#num_iters = 2000*6

num_iters = 100
learning_rate = 0.5*1e-4
iter_hook = IterHook()

# Build and train network
net1 = build_net(channel_size, num_layers, kernel_size).to(device)
net2 = build_net(channel_size, num_layers, kernel_size).to(device)
net3 = build_net(channel_size, num_layers, kernel_size).to(device)

net_path1 = "DL_Fit_Axial_Plane.pt"
net1.load_state_dict(torch.load(net_path1))

net_path2 = "DL_Fit_Coronal_Plane.pt"
net2.load_state_dict(torch.load(net_path2))

net_path3 = "DL_Fit_Sagittal_Plane.pt"
net3.load_state_dict(torch.load(net_path3))

net1, net2, net3 = train_net(
    net1, 
    net2, 
    net3,
    dataset, 
    num_iters, 
    learning_rate,
    learning_rate, 
    learning_rate, 
    
    kernel_size, 
    res, 
    muwf, 
    iter_hook, 
    show_progress=True,
    #device=device
)
losses = pd.DataFrame(iter_hook.losses) 

# Save losses and model
net1_path = "DL_Fit_Axial_Plane_Coupling.pt"
net2_path = "DL_Fit_Coronal_Plane_Coupling.pt"
net3_path = "DL_Fit_Sagittal_Plane_Coupling.pt"

torch.save(net1.state_dict(), net1_path)
torch.save(net2.state_dict(), net2_path)
torch.save(net3.state_dict(), net3_path)
