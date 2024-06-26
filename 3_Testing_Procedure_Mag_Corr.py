import scipy.io
import copy
from itertools import cycle
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
from skimage.metrics import normalized_root_mse as nrmse, structural_similarity as ssim, peak_signal_noise_ratio as psnr
from Util import configure
import math
from torch import nn
from Util_Mag_Corr_Joint import laplacian_fn, mag_weight_fn, neural_weight_fn, build_net, train_net

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
imgs_test, masks_test, b1_corr_test = [
    data_test[k] for k in ["dataset_training", "dataset_mask", "dataset_b1_corr"]
]
imgs_test, masks_test, b1_corr_test = (
    torch.tensor(imgs_test, dtype=torch.complex64),
    torch.tensor(masks_test, dtype=bool),
    torch.tensor(b1_corr, dtype=torch.float),
)
imgs_test, masks_test, b1_corr_test = [x.to(device) for x in [imgs_test, masks_test, b1_corr_test]]

phase_test = torch.angle(imgs_test)
mag_test = torch.abs(imgs_test)

# Test for Axial Plane
DL_result_axial = []
for idx in range(imgs_test.size(0)):
    imgs_idx = imgs_test[idx]
    mask_idx = mag_test[idx]
    phase_idx = torch.angle(imgs_idx)
    mag_idx = torch.abs(imgs_idx)
    
    lap_fn = laplacian_fn(mask_idx, kernel_size, res)
    mean_idx, std_idx = torch.mean(mag_idx), torch.std(mag_idx)
    
    with torch.no_grad():
        cond_nn = lap_fn(phase_idx /2, neural_weight_fn(mag_idx, mask_idx, net1, mean_idx, std_idx)) / muwf
        DL_result = torch.where(cond_nn > 0, cond_nn2, lap_fn(phase_idx / 2, mag_weight_fn(mag_idx, mask_idx, sigma=0.2)) / muwf)
        DL_result_axial.append(DL_result)
DL_result_axial = torch.stack(DL_result_axial) 
DL_result_axial = DL_result_axial + b1_corr_test

# Test for Coronal Plane
masks_test_coronal = masks_test.permute(1,0,2)
imgs_test_coronal = imgs_test.permute(1,0,2)

phase_test_coronal = torch.angle(imgs_test_coronal)
mag_test_coronal = torch.abs(imgs_test_coronal)

DL_result_coronal = []
DL_result = []
for idx in range(imgs_test_coronal.size(0)):
    imgs_idx_coronal = imgs_test_coronal[idx]
    mask_idx_coronal = mag_test_coronal[idx]
    phase_idx_coronal = torch.angle(imgs_idx_coronal)
    mag_idx_coronal = torch.abs(imgs_idx_coronal)
    
    lap_fn = laplacian_fn(mask_idx_coronal, kernel_size, res)
    mean_idx_coronal, std_idx_coronal = torch.mean(mag_idx_coronal), torch.std(mag_idx_coronal)
    
    with torch.no_grad():
        cond_nn_coronal = lap_fn(phase_idx_coronal /2, neural_weight_fn(mag_idx_coronal, mask_idx_coronal, net2, mean_idx_coronal, std_idx_coronal)) / muwf
        DL_result = torch.where(cond_nn > 0, cond_nn2_coronal, lap_fn(phase_idx_coronal / 2, mag_weight_fn(mag_idx_coronal, mask_idx_coronal, sigma=0.2)) / muwf)
        DL_result_coronal.append(DL_result)
DL_result_coronal = torch.stack(DL_result_coronal) 
DL_result_coronal = DL_result_coronal.permute(1,0,2)
DL_result_coronal = DL_result_coronal + b1_corr_test

# Test for Sagittal Plane
masks_test_sagittal = masks_test.permute(2,0,1)
imgs_test_sagittal = imgs_test.permute(2,0,1)

phase_test_sagittal = torch.angle(imgs_test_sagittal)
mag_test_sagittal = torch.abs(imgs_test_sagittal)

DL_result_sagittal = []
DL_result = []
for idx in range(imgs_test_sagittal.size(0)):
    imgs_idx_sagittal = imgs_test_sagittal[idx]
    mask_idx_sagittal = mag_test_sagittal[idx]
    phase_idx_sagittal = torch.angle(imgs_idx_sagittal)
    mag_idx_sagittal = torch.abs(imgs_idx_sagittal)
    
    lap_fn = laplacian_fn(mask_idx_sagittal, kernel_size, res)
    mean_idx_sagittal, std_idx_sagittal = torch.mean(mag_idx_sagittal), torch.std(mag_idx_sagittal)
    
    with torch.no_grad():
        cond_nn_sagittal = lap_fn(phase_idx_sagittal /2, neural_weight_fn(mag_idx_sagittal, mask_idx_sagittal, net3, mean_idx_sagittal, std_idx_sagittal)) / muwf
        DL_result = torch.where(cond_nn > 0, cond_nn_sagittal, lap_fn(phase_idx_sagittal / 2, mag_weight_fn(mag_idx_sagittal, mask_idx_sagittal, sigma=0.2)) / muwf)
        DL_result_sagittal.append(DL_result)
DL_result_sagittal = torch.stack(DL_result_sagittal) 
DL_result_sagittal = DL_result_sagittal.permute(1,2,0)
DL_result_sagittal = DL_result_sagittal + b1_corr_test

DL_result_coupled = (DL_result_axial + DL_result_coronal + DL_result_sagittal) / 3

# Save
DL_result_axial_np = DL_result_axial.cpu().numpy()
DL_result_coronal_np = DL_result_coronal.cpu().numpy()
DL_result_sagittal_np = DL_result_sagittal.cpu().numpy()
DL_result_coupled_np = DL_result_coupled.cpu().numpy()
b1_corr_test_np = b1_corr_test.numpy()
imgs_test_np = imgs_test.cpu().numpy()
masks_test_np =  masks_test.cpu().numpy()
b1_corr_test_np = b1_corr_test.numpy()

mat_dict = {
    'DL_axial': DL_result_axial_np, 
    'DL_coronal': DL_result_coronal_np, 
    'DL_sagittal': DL_result_sagittal_np,
    'DL_coupled': DL_result_coupled_np,
    'imgs': imgs_test_np, 
    'mask': masks_test_np,
    'b1_corr': b1_corr_test_np,
}
scipy.io.savemat('Result.mat', mat_dict)
