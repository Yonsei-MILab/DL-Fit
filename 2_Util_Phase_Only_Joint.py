import math
from itertools import chain, combinations, islice
import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from math import exp
from copy import deepcopy
from torch.utils.data import DataLoader

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
    net1, 
    net2, 
    net3,
    dataset,
    num_iters,
    learning_rate1, 
    learning_rate2,
    learning_rate3,
    kernel_size,
    res,
    muwf,
    iter_hook=lambda i, loss: None,
    show_progress=True,
    device=None,
):
    dataset = islice(dataset, num_iters)
    optimizer1 = torch.optim.Adam(params=net1.parameters(), lr=learning_rate1)
    optimizer2 = torch.optim.Adam(params=net2.parameters(), lr=learning_rate2)
    optimizer3 = torch.optim.Adam(params=net3.parameters(), lr=learning_rate3)
    criterion = nn.MSELoss()

    pbar = tqdm(enumerate(dataset, 1), total=num_iters, disable=not show_progress)
    for i, (imgs, mask, gt) in pbar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        imgs_noise = add_noise_to_complex(imgs, mask)
        phases, mags = torch.angle(imgs_noise), torch.abs(imgs_noise)
        mean, std = torch.mean(mags), torch.std(mags)

        all_cond_net1 = []
        all_cond_net2 = []
        all_cond_net3 = []

        # Axial Plane
        for j in range(imgs_noise.shape[0]):
            cur_phases, cur_mags = phases[j].squeeze(0), mags[j].squeeze(0)
            cur_mask = mask[j].squeeze(0)
            weight_fn_net1 = neural_weight_fn(cur_mags, cur_mask, net1, mean, std)
            lap_fn_net1 = laplacian_fn(cur_mask, kernel_size, res)
            cond_net1 = lap_fn_net1(cur_phases / 2, weight_fn_net1) / muwf
            if torch.isnan(cond_net1).any() or torch.isinf(cond_net1).any():
                continue
            all_cond_net1.append(cond_net1.unsqueeze(0))

        # Sagittal
        masks_zy = mask.permute(2, 0, 1)
        mags_zy = mags.permute(2, 0, 1)
        phases_zy = phases.permute(2, 0, 1)
        gt_zy = gt.permute(2, 0, 1)

        for j in range(masks_zx.shape[0]):
            cur_phases_zy, cur_mags_zy = phases_zy[j].squeeze(0), mags_zy[j].squeeze(0)
            cur_mask_zy = masks_zy[j].squeeze(0)
            #cur_gt_zy = gt_zy[j].squeeze(0)

            weight_fn_net2 = neural_weight_fn(cur_mags_zy, cur_mask_zy, net2, mean, std)
            lap_fn_net2 = laplacian_fn(cur_mask_zy, kernel_size, res)
            cond_net2 = lap_fn_net2(cur_phases_zy / 2, weight_fn_net2) / muwf
            if torch.isnan(cond_net2).any() or torch.isinf(cond_net2).any():
                continue
            all_cond_net2.append(cond_net2.unsqueeze(0))  # 3D 텐서로 쌓기 위해 차원 추가

        # Coronal
        masks_zy = mask.permute(1, 2, 0)
        mags_zy = mags.permute(1, 2, 0)
        phases_zy = phases.permute(1, 2, 0)
        gt_zy = gt.permute(1, 2, 0)

        for j in range(masks_zy.shape[0]):
            cur_phases_zy, cur_mags_zy = phases_zy[j].squeeze(0), mags_zy[j].squeeze(0)
            cur_mask_zy = masks_zy[j].squeeze(0)
            #cur_gt_zx = gt_zx[j].squeeze(0)

            weight_fn_net3 = neural_weight_fn(cur_mags_zy, cur_mask_zy, net3, mean, std)
            lap_fn_net3 = laplacian_fn(cur_mask_zy, kernel_size, res)
            cond_net3 = lap_fn_net3(cur_phases_zy / 2, weight_fn_net3) / muwf
            # NaN 또는 inf 검사를 계속 수행합니다.
            if torch.isnan(cond_net3).any() or torch.isinf(cond_net3).any():
                continue
            all_cond_net3.append(cond_net3.unsqueeze(0))  # 3D 텐서로 쌓기 위해 차원 추가

        # 결과 쌓기
        if all_cond_net1 and all_cond_net2 and all_cond_net3:  # 리스트가 비어있지 않은 경우에만 진행
            all_cond_net1 = torch.cat(all_cond_net1, dim=0)  # 3D 텐서 생성
            all_cond_net2 = torch.cat(all_cond_net2, dim=0)  # 3D 텐서 생성
            all_cond_net3 = torch.cat(all_cond_net3, dim=0) 

            # 손실 계산
            loss_net1 = criterion(all_cond_net1, gt)  # 여기서 `gt`는 net1의 ground truth 여야 합니다.
            loss_net1_ssim = 0.5*(1 - ssim(all_cond_net1.unsqueeze(0), gt.unsqueeze(0)))
            loss_net2 = criterion(all_cond_net2, gt_zx)  # 여기서 `gt`는 net2의 ground truth 여야 합니다.
            loss_net2_ssim = 0.5*(1 - ssim(all_cond_net2.unsqueeze(0), gt_zx.unsqueeze(0)))
            loss_net3 = criterion(all_cond_net3, gt_zy)   
            loss_net3_ssim = 0.5*(1 - ssim(all_cond_net3.unsqueeze(0), gt_zy.unsqueeze(0)))
              
            all_cond_net2 = all_cond_net2.permute(1, 2, 0)
            all_cond_net3 = all_cond_net3.permute(2, 0, 1)
            
            tri_cond =  (all_cond_net1+all_cond_net2 +all_cond_net3)/3
            loss_tri = criterion(tri_cond, gt)
            loss_tri_ssim = 0.5*(1 - ssim(tri_cond.unsqueeze(0), gt.unsqueeze(0)))

            total_loss = 3*(loss_tri+loss_tri_ssim) #+ (loss_net1+loss_net1_ssim + loss_net2 +loss_net2_ssim+ loss_net3 + loss_net3_ssim)  # 두 네트워크의 평균 손실
            total_loss.backward()  # 역전파
            optimizer1.step()  # 네트워크1 업데이트
            optimizer2.step()  # 네트워크2 업데이트
            optimizer3.step()

            pbar.set_description(f"loss={total_loss.item():.4f}")
            iter_hook(i, total_loss.item())


    # 학습이 끝난 네트워크 반환
    return net1, net2  , net3     
   

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

