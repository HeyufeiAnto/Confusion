import torch
import numpy as np
from net import Encoder, Decoder, Branch_Encoder_Ec, Branch_Encoder_Exi, Branch_Encoder_Exv, SF_Fuse, CF_Fuse
from einops import rearrange
import cv2
from torch.nn.functional import interpolate
from mask_generator1 import GroundedSAM2MaskGenerator

import os
import json
import time
from model import longclip

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

import torch.nn as nn
from torch.utils.data import DataLoader
from loss import rgb_to_ycbcr, ycbcr_to_rgb


# ============================================================
# PATHS
# ============================================================

ckpt_path = "" # weight of training stage 2
path_ir = ''
path_vi = ''
save_fuse_path = ''

os.makedirs(save_fuse_path, exist_ok=True)

img_name = os.listdir(path_ir)
img_name = sorted(img_name)


# ============================================================
# DEVICE
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ============================================================
# LOAD MODELS
# ============================================================

gen = GroundedSAM2MaskGenerator()

En = nn.DataParallel(Encoder()).to(device).float()
Ec = nn.DataParallel(Branch_Encoder_Ec()).to(device).float()
Ex_v = nn.DataParallel(Branch_Encoder_Exv()).to(device).float()
Ex_i = nn.DataParallel(Branch_Encoder_Exi()).to(device).float()
Dec = nn.DataParallel(Decoder()).to(device).float()
S_Fuse = nn.DataParallel(SF_Fuse()).to(device).float()
C_Fuse = nn.DataParallel(CF_Fuse()).to(device).float()

model_clip, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

ckpt = torch.load(ckpt_path, map_location=device)

En.load_state_dict(ckpt['En'])
Ec.load_state_dict(ckpt['Ec'])
Ex_v.load_state_dict(ckpt['Ex_vis'])
Ex_i.load_state_dict(ckpt['Ex_ir'])
Dec.load_state_dict(ckpt['Dec'])
S_Fuse.load_state_dict(ckpt['Sfuse'])
C_Fuse.load_state_dict(ckpt['Cfuse'])

cat = ("person", "car", "truck", "motor", "lamp", "bus")
cat_num = len(cat)

En.eval()
Ec.eval()
Ex_v.eval()
Ex_i.eval()
Dec.eval()
S_Fuse.eval()
C_Fuse.eval()
model_clip.eval()

fusion_modules = {
    "Encoder": En,
    "Branch_Encoder_Ec": Ec,
    "Branch_Encoder_Exv": Ex_v,
    "Branch_Encoder_Exi": Ex_i,
    "Decoder": Dec,
    "SF_Fuse": S_Fuse,
    "CF_Fuse": C_Fuse,
}

with torch.no_grad():
    for idx, name in enumerate(img_name):
        print(f"\n[{idx + 1}/{len(img_name)}] Processing: {name}")

        img_IR = cv2.imread(path_ir + '/' + name)
        img_VI = cv2.imread(path_vi + '/' + name)

        if img_IR is None or img_VI is None:
            print(f"[Warning] Failed to read image: {name}")
            continue

        prompt = longclip.tokenize(cat).to(device)
        text_f = model_clip.encode_text(prompt)
        text_f = text_f / text_f.norm(dim=1, keepdim=True)
        text_f = text_f.unsqueeze(0).expand(1, cat_num, 512)

        # This version allows users to directly assign the alpha without LLM.
        alpha_dict = {
            'person': 1.0,
            'car': 0.5,
            'background': 0.5,
            'bus': 0.5,
            'truck': 0.5
        }
        maskt = torch.from_numpy(maskt_numpy).float().to(device)

        img_VI = cv2.cvtColor(img_VI, cv2.COLOR_BGR2RGB)

        img_IR = torch.from_numpy(img_IR).float() / 255.0
        img_VI = torch.from_numpy(img_VI).float() / 255.0

        img_IR = img_IR.unsqueeze(0)
        img_VI = img_VI.unsqueeze(0)

        img_IR = rearrange(img_IR, 'b h w c -> b c h w').to(device)
        img_VI = rearrange(img_VI, 'b h w c -> b c h w').to(device)

        img_VI = rgb_to_ycbcr(img_VI)

        y = img_VI[:, 0:1, :, :]
        cb = img_VI[:, 1:2, :, :]
        cr = img_VI[:, 2:, :, :]

        img_IR = img_IR[:, 0:1, :, :]

        ones = torch.ones_like(maskt)
        maskb = ones - maskt

        vi_f = En(y)
        ir_f = En(img_IR)

        vi_c_f = Ec(vi_f)
        vi_s_f = Ex_v(vi_f)

        ir_c_f = Ec(ir_f)
        ir_s_f = Ex_i(ir_f)

        f_c_f = C_Fuse(ir_c_f, vi_c_f, text_f)
        f_s_f = S_Fuse(ir_s_f, vi_s_f, maskt, maskb)

        fuseimg = Dec(f_c_f, f_s_f)

        img_F = torch.cat([fuseimg, cb, cr], dim=1)
        img_F = ycbcr_to_rgb(img_F)

        img_F = rearrange(img_F, 'b c h w -> b h w c')
        img_F = torch.squeeze(img_F, 0)
        img_F = img_F.cpu()
        img_F = (img_F.numpy() * 255.0).astype(np.uint8)

        img_F = cv2.cvtColor(img_F, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(save_fuse_path, name)
        cv2.imwrite(save_path, img_F)
