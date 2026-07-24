# -*- coding: utf-8 -*-
from PIL.Image import item
# from lang_sam import LangSAM
from kornia.losses import ssim
from kornia.losses.ssim import SSIMLoss
from net import Encoder, Decoder, Branch_Encoder_Ec, Branch_Encoder_Exi, Branch_Encoder_Exv, Text_project
# from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from utils.loss import Fusionloss
from loss import rgb_to_ycbcr, Loss_cov, Label_loss
import kornia
from dataload1 import data_made, Getloader
from einops import rearrange
from model import longclip
from torch.nn.functional import interpolate, l1_loss
import numpy as np
import sys

path_VI = ''
path_IR = ''
path_mask = ''              # path to Rj(x,y)
cat = ("person", "car", 'truck', 'motor', 'lamp', 'bus')
cat_num = len(cat)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# . Set the hyper-parameters for training
num_epochs = 10  # total epoch

lr = 1e-4
weight_decay = 0
batch_size = 2
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

clip_grad_norm_value = 0.01
optim_step = 10
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

En = nn.DataParallel(Encoder()).to(device)
Ec = nn.DataParallel(Branch_Encoder_Ec()).to(device)

Ex_v = nn.DataParallel(Branch_Encoder_Exv()).to(device)
Ex_i = nn.DataParallel(Branch_Encoder_Exi()).to(device)

Dec = nn.DataParallel(Decoder()).to(device)

Tpr = nn.DataParallel(Text_project()).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    En.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    Ec.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    Ex_v.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    Ex_i.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    Dec.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(
    Tpr.parameters(), lr=lr, weight_decay=weight_decay)


scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)


MSELoss = nn.MSELoss()
Loss_ssim = kornia.losses.SSIMLoss(11, eps=1e-6, reduction='mean')
criteria_fusion = Fusionloss()
L1_loss = nn.L1Loss()
labelloss = Label_loss()


# data loader
image_patches = data_made(path_VI, path_IR, path_mask, cat)
datasets = Getloader(image_patches)
trainloader = torch.utils.data.DataLoader(datasets, batch_size=2, shuffle=True, drop_last=False, num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# Train
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
Lossbest = 10000
count = 0

model_clip, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)
logit_scale = model_clip.logit_scale.exp()

for epoch in range(num_epochs):
    count += 1
    for i, data in enumerate(loader['train']):
        data_VIS, data_IR, data_Masks, cat = data
    
        # 移至 GPU
        data_VIS = data_VIS.to(device)
        data_IR = data_IR.to(device)
        data_Mask = data_Masks.to(device)

        data_VIS = rgb_to_ycbcr(data_VIS)
        y = data_VIS[:, 0:1, :, :]
        data_IR = data_IR[:, 0:1, :, :]

        if isinstance(cat[0], tuple):
            cat = [t[0] for t in cat]
        
        size = data_IR.shape[0]

        prompt = longclip.tokenize(cat).to(device)
        text_f = model_clip.encode_text(prompt)
        text_f = text_f / text_f.norm(dim=1, keepdim=True)
        text_f = text_f.unsqueeze(0).expand(size, cat_num, 512)
 
        En.train()
        Ec.train()
        Ex_v.train()
        Ex_i.train()
        Dec.train()
        Tpr.train()

        En.zero_grad()
        Ec.zero_grad()
        Ex_v.zero_grad()
        Ex_i.zero_grad()
        Dec.zero_grad()
        Tpr.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()

        vi_f = En(y)
        ir_f = En(data_IR)
        vi_c_f = Ec(vi_f)
        vi_s_f = Ex_v(vi_f)
        ir_c_f  = Ec(ir_f)
        ir_s_f = Ex_i(ir_f)

        
        vi_r = Dec(vi_c_f, vi_s_f)
        ir_r = Dec(ir_c_f, ir_s_f)
        vi_i_r = Dec(ir_c_f, vi_s_f)
        ir_v_r = Dec(vi_c_f, ir_s_f)
        

        loss_label = torch.tensor(0.0, device=device)
        for k in range(size):
            for m in range(cat_num):
                sum = torch.sum(data_Mask[k:(k+1), m:(m+1), :, :], dim=(2, 3))
                has0 = torch.any(torch.abs(sum) < 1e-6)
                prompt = text_f[k:(k+1), m:(m+1), :]
                if has0:
                    loss_label += 0
                else:
                    labelv = vi_c_f[k:(k+1), :, :, :] * data_Mask[k:(k+1), m:(m+1), :, :]
                    labelv = Tpr(labelv, data_Mask[k:(k+1), m:(m+1), :, :])
                    labelv = labelv.unsqueeze(1)
                    labeli = ir_c_f[k:(k+1), :, :, :] * data_Mask[k:(k+1), m:(m+1), :, :]
                    labeli = Tpr(labeli, data_Mask[k:(k+1), m:(m+1), :, :])
                    labeli = labeli.unsqueeze(1)
                    loss_label = loss_label + labelloss(labelv, prompt) + labelloss(labeli, prompt)

        loss_label = loss_label / (size * 2 * cat_num)


        loss_cc = MSELoss(vi_c_f, ir_c_f) 
        loss_ce = Loss_cov(vi_c_f, vi_s_f) + Loss_cov(ir_c_f, ir_s_f)
        loss_rec = MSELoss(y, vi_r) + MSELoss(data_IR, ir_r) + Loss_ssim(y, vi_r) + Loss_ssim(data_IR, ir_r)
        loss_cross = MSELoss(y, vi_i_r) + MSELoss(data_IR, ir_v_r) + Loss_ssim(y, vi_i_r) + Loss_ssim(data_IR, ir_v_r)
        loss_grad = L1_loss(kornia.filters.SpatialGradient()(y),
                                       kornia.filters.SpatialGradient()(vi_r))
        
        loss= loss_rec + loss_grad * 5 + loss_cross + loss_cc + 5 * loss_ce + loss_label * 0.2
        
        loss.backward()
        nn.utils.clip_grad_norm_(
            En.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            Ec.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            Ex_v.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            Ex_i.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            Dec.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            Tpr.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()
        optimizer6.step()
        
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [total_loss: %f] [rec_loss: %f] [cross_loss: %f] [grad_loss: %f] [cc_loss: %f] [ce_loss: %f]  ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                loss_rec.item(),
                loss_cross.item(),
                loss_grad.item(),
                loss_cc.item(),
                loss_ce.item(),
                time_left,
            )
            )
    # adjust the learning rate
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()
    scheduler6.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6


if True:
    checkpoint = {
        'En': En.state_dict(),
        'Ec': Ec.state_dict(),
        'Ex_vis': Ex_v.state_dict(),
        'Ex_ir': Ex_i.state_dict(),
        'Dec': Dec.state_dict(),
        'Tpr': Tpr.state_dict(),
        }
    torch.save(checkpoint, os.path.join())
