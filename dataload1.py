from torch.utils.data import DataLoader
import os
import cv2
import torch
import numpy as np

def data_made(vis_path, ir_path, mask_root_path, catagories, target_size=(240, 240)):
    name_vis = sorted(os.listdir(vis_path))
    name_ir = sorted(os.listdir(ir_path))
    name = zip(name_vis, name_ir)
    datasets = []

    for vis, ir in name:
        if vis != ir:
            print('image is not matched')
            break
        imgv = cv2.imread(os.path.join(vis_path, vis))
        imgi = cv2.imread(os.path.join(ir_path, ir))
        imgv = cv2.cvtColor(imgv, cv2.COLOR_BGR2RGB)
        # 如果红外是单通道，保持原样；如果是3通道BGR，也转RGB
        if len(imgi.shape) == 3:
             imgi = cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        imgv = cv2.resize(imgv, target_size)
        imgi = cv2.resize(imgi, target_size)

        mask_list = []
        for target_obj in catagories:
            # 构建 Mask 路径
            mask_name = vis
            mask_file = os.path.join(mask_root_path, target_obj, mask_name)
            
            # 读取 Mask
            if os.path.exists(mask_file):
                imgmask = cv2.imread(mask_file, 0) # 灰度读取
                imgmask = cv2.resize(imgmask, target_size, interpolation=cv2.INTER_NEAREST)
            else:
                imgmask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
            
            mask_list.append(imgmask)
        
        # 堆叠 Mask: (6, 320, 320)
        masks_np = np.stack(mask_list, axis=0)

        # 存入 List
        # 注意：不要在这一步做 expand_dims 或 concat，保持结构清晰
        datasets.append([imgv, imgi, masks_np, catagories])
    
    
    return datasets


class Getloader(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        self.data = dataroot
    
    def __getitem__(self, index):
        # 解包数据
        vis, ir, mask, text = self.data[index]
        
        # 1. 图像转 Tensor: (H,W,C) -> (C,H,W), 归一化到 0-1
        # 必须转为 float Tensor，否则 collate 可能报错
        vis = torch.from_numpy(vis.transpose(2, 0, 1).copy()).float() / 255.0
        
        # 红外图处理
        if len(ir.shape) == 2: # 如果是单通道 (H, W)
            ir = torch.from_numpy(ir.copy()).float().unsqueeze(0) / 255.0
        else: # 如果是三通道 (H, W, 3)
            ir = torch.from_numpy(ir.transpose(2, 0, 1).copy()).float() / 255.0
            
        # 2. Mask 转 Tensor
        # mask 已经是 (N_class, H, W) 了，直接转
        mask = torch.from_numpy(mask.copy()).float() / 255.0
        
        # text 是列表 ['car', 'person'...]，DataLoader 会自动处理成列表的列表
        
        return vis, ir, mask, text

    def __len__(self):
        return len(self.data)
