from torch.utils.data import DataLoader
import os
import cv2
import torch
import numpy as np

def data_made(vis_path, ir_path, seg_path, bina_path, catgories, target_size=(240, 240)):
    name_vis = sorted(os.listdir(vis_path))
    name_ir = sorted(os.listdir(ir_path))
    name_seg = sorted(os.listdir(seg_path))
    name_bina = sorted(os.listdir(bina_path))
    name = zip(name_vis, name_ir, name_seg, name_bina)
    datasets = []

    for vis, ir, seg, bina in name:
        if vis != ir or vis != seg or ir != seg or vis != bina or ir != bina or seg != bina:
            print('image is not matched')
            break
        imgv = cv2.imread(os.path.join(vis_path, vis))
        imgi = cv2.imread(os.path.join(ir_path, ir))
        imgseg = cv2.imread(os.path.join(seg_path, seg))
        imgbina = cv2.imread(os.path.join(bina_path, bina))
        imgv = cv2.cvtColor(imgv, cv2.COLOR_BGR2RGB)
        # 如果红外是单通道，保持原样；如果是3通道BGR，也转RGB
        if len(imgi.shape) == 3:
             imgi = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB)
        if len(imgseg.shape) == 3:
             imgseg = cv2.cvtColor(imgseg, cv2.COLOR_BGR2RGB)
        if len(imgbina.shape) == 3:
             imgbina = cv2.cvtColor(imgbina, cv2.COLOR_BGR2RGB)
        # 调整图像大小
        imgv = cv2.resize(imgv, target_size)
        imgi = cv2.resize(imgi, target_size)
        imgseg = cv2.resize(imgseg, target_size, interpolation=cv2.INTER_NEAREST)
        imgbina = cv2.resize(imgbina, target_size, interpolation=cv2.INTER_NEAREST)
        
        datasets.append([imgv, imgi, imgseg, imgbina, catgories])
    
    return datasets


class Getloader(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        self.data = dataroot
    
    def __getitem__(self, index):
        # 解包数据
        vis, ir, seg, bina, text = self.data[index]
        
        # 1. 图像转 Tensor: (H,W,C) -> (C,H,W), 归一化到 0-1
        # 必须转为 float Tensor，否则 collate 可能报错
        vis = torch.from_numpy(vis.transpose(2, 0, 1).copy()).float() / 255.0
        
        # 红外图处理
        if len(ir.shape) == 2: # 如果是单通道 (H, W)
            ir = torch.from_numpy(ir.copy()).float().unsqueeze(0) / 255.0
        else: # 如果是三通道 (H, W, 3)
            ir = torch.from_numpy(ir.transpose(2, 0, 1).copy()).float() / 255.0
        
        if len(seg.shape) == 2: # 如果是单通道 (H, W)
            seg = torch.from_numpy(seg.copy()).float().unsqueeze(0) / 255.0
        else: # 如果是三通道 (H, W, 3)
            seg = torch.from_numpy(seg.transpose(2, 0, 1).copy()).float() / 255.0
        
        if len(bina.shape) == 2: # 如果是单通道 (H, W)
            bina = torch.from_numpy(bina.copy()).float().unsqueeze(0) / 255.0
        else: # 如果是三通道 (H, W, 3)
            bina = torch.from_numpy(bina.transpose(2, 0, 1).copy()).float() / 255.0
            
        return vis, ir, seg, bina, text

    def __len__(self):
        return len(self.data)
