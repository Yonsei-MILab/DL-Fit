import math
from itertools import chain, combinations, islice
import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from math import exp
from copy import deepcopy

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

def same(kernel_size):
    return (kernel_size - 1) // 2, kernel_size // 2

def build_win(mask, kernel_size):
    assert mask.dim() == len(kernel_size)
    padding = [same(ks) for ks in kernel_size]
    half, _ = zip(*padding)
    mask = nn.functional.pad(mask, list(reversed(list(chain.from_iterable(padding)))))
    maskidx = mask.nonzero()
    revidx = torch.zeros_like(mask, dtype=maskidx.dtype)
    revidx[mask] = torch.arange(maskidx.size(0), device=mask.device)
    half = maskidx.new_tensor(half).reshape(1, 1, -1)
    win = mask.new_ones(*kernel_size).nonzero() - half
    idx = maskidx.unsqueeze(1) + win
    idx = idx.unbind(-1)
    winmask, winidx = mask[idx], revidx[idx]
    return winmask, winidx

def build_vander(mask, res):
    assert mask.dim() == len(res)
    cords = mask.nonzero(as_tuple=True)
    cords = [c * r for c, r in zip(cords, res)]
    vander = [torch.ones_like(cords[0])]
    lmbda = [0]
    for cord in cords:
        vander += [cord]
        lmbda += [0]
    for cord1, cord2 in combinations(cords, 2):
        vander += [cord1 * cord2]
        lmbda += [0]
    for cord in cords:
        vander += [cord ** 2]
        lmbda += [2]
    vander = torch.stack(vander, -1)
    lmbda = vander.new_tensor(lmbda).unsqueeze(0)
    return vander, lmbda

def calc_lstsq(vander, target, winmask, winidx, weight):
    vander = vander[winidx]
    target = target[winidx].unsqueeze(-1)
    mult = (winmask * weight).unsqueeze(-1)
    q, r = torch.linalg.qr(mult * vander)
    try:
        coeffs = torch.linalg.pinv(r) @ q.transpose(1, 2) @ (mult * target)
    except:
        coeffs = torch.ones_like(r @ q.transpose(1, 2) @ (mult * target))
    return coeffs

def default_weight_fn():
    def fn(winidx):
        return torch.ones_like(winidx)
    return fn

def mag_weight_fn(mag, mask, sigma):
    mag = mag[mask]
    mag = mag / torch.std(mag)
    def fn(winidx):
        dist = mag[winidx] - mag.unsqueeze(-1)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    return fn

def build_net(channel_size, num_layers, kernel_size):
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

def add_noise(x):
    noise = 0.006 * torch.randn(x.shape).cuda()
    return x + noise

def add_noise_to_complex(x, y):
    z = (add_noise(torch.real(x).cuda()) + 1j * add_noise(torch.imag(x).cuda())) * y
    return z

def train_net(
    net,
    dataset,
    num_iters,
    learning_rate,
    kernel_size,
    res,
    muwf,
    iter_hook=lambda i, loss: None,
    show_progress=True,
):
    dataset = islice(dataset, num_iters)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
    
    pbar = tqdm(enumerate(dataset, 1), total=num_iters, disable=not show_progress)
    for i, (imgs, mask, gt) in pbar:    
        optimizer.zero_grad()
        lap_fn = laplacian_fn(mask, kernel_size, res)
        imgs_noise = add_noise_to_complex(imgs, mask)
        phases, mags = torch.angle(imgs_noise), torch.abs(imgs_noise)
        mean, std = torch.mean(mags), torch.std(mags)
        cond = (lap_fn(phases / 2, neural_weight_fn(mags, mask, net, mean, std)) / muwf)
        loss = (nn.functional.mse_loss(cond[mask], gt[mask])  + 0.5 * (1 - ssim(torch.unsqueeze(torch.unsqueeze(cond, 0), 0), torch.unsqueeze(torch.unsqueeze(gt, 0), 0))))

        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        optimizer.step()       
        loss = loss.item()
        pbar.set_description(f"loss={loss:.4f}")
        iter_hook(i, loss)
    return net

def neural_weight_fn(mag, mask, net, mean, std):
    sigma = 0.2
    mag = mag[mask]
    mag = (mag - mean) / std
    def fn(winidx):
        dist = mag[winidx] - mag.unsqueeze(-1)
        return net(torch.exp(-(dist ** 2) / (2 * sigma ** 2)))
    return fn

def laplacian_fn(mask, kernel_size, res):
    winmask, winidx = build_win(mask, kernel_size)
    vander, lmbda = build_vander(mask, res)
    def fn(phase, weight_fn=default_weight_fn()):
        weight = weight_fn(winidx)
        target = phase[mask]
        coeffs = calc_lstsq(vander, target, winmask, winidx, weight)
        lap = lmbda.unsqueeze(0) @ coeffs
        lap = torch.zeros_like(phase).masked_scatter_(mask.contiguous(), lap)
        return lap
    return fn

