from PIL.Image import item
from kornia.losses import ssim
from kornia.losses.ssim import SSIMLoss
# from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from utils.loss import Fusionloss
from net import Encoder, Decoder, Branch_Encoder_Ec, Branch_Encoder_Exi, Branch_Encoder_Exv, SF_Fuse, CF_Fuse
from loss import rgb_to_ycbcr, Grad_extract
import kornia
from dataload2 import data_made, Getloader
from einops import rearrange
from model import longclip
from torch.nn.functional import interpolate
import numpy as np
import sys

ckpt_path = ""  # weight of training stage 1

path_VI = ''
path_IR = ''
path_Bina = ''   # path to binary masks
path_Modulation = '' # path to modulation masks

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# . Set the hyper-parameters for training
num_epochs = 50  # total epoch

lr = 1 * 1e-4
lr2 = 2.5 * 1e-5
weight_decay = 0
batch_size = 2
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

clip_grad_norm_value = 0.01
optim_step = 10
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
En = nn.DataParallel(Encoder()).to(device).float()
Ec = nn.DataParallel(Branch_Encoder_Ec()).to(device).float()
Ex_v = nn.DataParallel(Branch_Encoder_Exv()).to(device).float()
Ex_i = nn.DataParallel(Branch_Encoder_Exi()).to(device).float()
Dec = nn.DataParallel(Decoder()).to(device).float()
S_Fuse = nn.DataParallel(SF_Fuse()).to(device).float()
C_Fuse = nn.DataParallel(CF_Fuse()).to(device).float()

En.load_state_dict(torch.load(ckpt_path)['En'])
Ec.load_state_dict(torch.load(ckpt_path)['Ec'])
Ex_v.load_state_dict(torch.load(ckpt_path)['Ex_vis'])
Ex_i.load_state_dict(torch.load(ckpt_path)['Ex_ir'])
Dec.load_state_dict(torch.load(ckpt_path)['Dec'])

cat = ("person", "car", 'truck', 'motor', 'lamp', 'bus')

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    Dec.parameters(), lr=lr2, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    S_Fuse.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    C_Fuse.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss(reduction='none')
L1Loss = nn.L1Loss(reduction='none')
Loss_ssim = kornia.losses.SSIMLoss(11, eps=1e-6, reduction='none')
criteria_fusion = Fusionloss()
Gr = Grad_extract()

# data loader
image_patches = data_made(path_VI, path_IR, path_Modulation, path_Bina, cat)
datasets = Getloader(image_patches)
trainloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# Train
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

cat_num = len(cat)

model_clip, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)
logit_scale = model_clip.logit_scale.exp()

loss_best = 1000000000
count = 0

for epoch in range(num_epochs):
    count += 1
    for i, data in enumerate(loader['train']):
        data_VIS, data_IR, data_Mask, data_Bina, cat = data
    
        # 移至 GPU
        data_VIS = data_VIS.to(device)
        data_IR = data_IR.to(device)
        data_Mask = data_Mask.to(device)
        data_Bina = data_Bina.to(device)
        
        data_VIS = rgb_to_ycbcr(data_VIS)
        y = data_VIS[:, 0:1, :, :]
        data_IR = data_IR[:, 0:1, :, :]
        data_Mask = data_Mask[:, 0:1, :, :]
        data_Bina = data_Bina[:, 0:1, :, :]
        
        if isinstance(cat[0], tuple):
            cat = [t[0] for t in cat]
        
        size = data_IR.shape[0]

        prompt = longclip.tokenize(cat).to(device)
        text_f = model_clip.encode_text(prompt)
        text_f = text_f / text_f.norm(dim=1, keepdim=True)
        text_f = text_f.unsqueeze(0).expand(size, cat_num, 512)

        En.eval()
        Ec.eval()
        Ex_v.eval()
        Ex_i.eval()

        Dec.train()
        S_Fuse.train()
        C_Fuse.train()
        Dec.zero_grad()
        S_Fuse.zero_grad()
        C_Fuse.zero_grad()
       

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        maskt = data_Mask + (1 - data_Bina) * 0.5

        ones = torch.ones_like(maskt)
        maskb = ones - maskt

        vi_f = En(y)
        ir_f = En(data_IR)
        vi_c_f = Ec(vi_f)
        vi_s_f = Ex_v(vi_f)
        ir_c_f  = Ec(ir_f)
        ir_s_f = Ex_i(ir_f)

        f_c_f = C_Fuse(ir_c_f, vi_c_f, text_f)
        f_s_f = S_Fuse(ir_s_f, vi_s_f, maskt, maskb)
        fuseimg = Dec(f_c_f, f_s_f)

        # _, _, grad_loss = criteria_fusion(y, data_IR, fuseimg)
        grad_ir = Gr(data_IR)
        grad_vi = Gr(y)
        grad_fuse = Gr(fuseimg)

        mse_map_t = MSELoss(maskt * data_Bina * fuseimg, maskt * data_Bina * data_IR) + MSELoss(maskb * data_Bina * fuseimg, maskb * data_Bina * y)
        mse_map_b = MSELoss((1 - data_Bina) * fuseimg, (1 - data_Bina) * y) + MSELoss((1 - data_Bina) * fuseimg, (1 -data_Bina) * data_IR)

        grad_map_t = L1Loss(torch.max(data_Bina * grad_ir, data_Bina * grad_vi), data_Bina * grad_fuse)
        grad_map_b = L1Loss((1 - data_Bina) * torch.max(grad_vi, grad_ir), (1 - data_Bina) * grad_fuse)

        mse_map_t = torch.sum(mse_map_t, dim=(1, 2, 3)) / (torch.sum(data_Bina, dim=(1,2,3)) + 1e-8)
        mse_map_b = torch.sum(mse_map_b, dim=(1, 2, 3)) / (torch.sum((1 - data_Bina), dim=(1,2,3)) + 1e-8)

        grad_map_t = torch.sum(grad_map_t, dim=(1, 2, 3)) / (torch.sum(data_Bina, dim=(1,2,3)) + 1e-8)
        grad_map_b = torch.sum(grad_map_b, dim=(1, 2, 3)) / (torch.sum((1 - data_Bina), dim=(1,2,3)) + 1e-8)

        loss_mse = mse_map_t.mean() + mse_map_b.mean()
        loss_grad = grad_map_t.mean() * 1 + grad_map_b.mean() * 4
        loss = loss_mse + loss_grad 

        if loss < loss_best:
            loss_best = loss
            count = 0
            checkpoint = {
                'En': En.state_dict(),
                'Ec': Ec.state_dict(),
                'Ex_vis': Ex_v.state_dict(),
                'Ex_ir': Ex_i.state_dict(),
                'Dec': Dec.state_dict(),
                'Sfuse': S_Fuse.state_dict(),
                'Cfuse': C_Fuse.state_dict(),
            }
            torch.save(checkpoint, os.path.join("."))

        loss.backward()
        
        nn.utils.clip_grad_norm_(
            Dec.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            S_Fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            C_Fuse.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [MSEloss: %f] [gradloss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                loss_mse.item(),
                loss_grad.item(),
                time_left,
            )
            )
    # adjust the learning rate
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6

    if count >= 100:
        print('end')
        sys.exit()
    
    