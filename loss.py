import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import math
import numpy as np
import torch.nn as nn

class Label_loss(torch.nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, x1, x2):
        # 确保输入形状相同
        assert x1.shape == x2.shape, f"Shape mismatch: {x1.shape} vs {x2.shape}"
        
        # 沿特征维度计算余弦相似度，结果形状: (b, c)
        cos_sim = F.cosine_similarity(x1, x2, dim=2, eps=self.eps)
        
        # 将余弦相似度转换为余弦距离 (范围: 0-2)
        # 通常使用 1 - cos_sim，使相同向量距离为0，相反向量距离为2
        cos_dist = 1.0 - cos_sim
        
        # 根据reduction参数聚合
        if self.reduction == 'mean':
            return cos_dist.mean()
        elif self.reduction == 'sum':
            return cos_dist.sum()
        elif self.reduction == 'none':
            return cos_dist
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

def _sample_hw(feat, num=2048):
    """
    feat: (B, C, H, W)
    return: (S, C)  S = num spatial samples
    """
    B, C, H, W = feat.shape
    x = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    N = x.size(0)

    if num is None or num >= N:
        return x

    idx = torch.randint(0, N, (num,), device=feat.device)
    return x[idx]


def _standardize(x, eps=1e-6):
    """
    x: (S, C)
    """
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, keepdim=True).clamp_min(eps)
    return x

def Loss_cov(c_feat, s_feat, num_samples=2048):
    """
    c_feat, s_feat: (B, 64, 240, 240)
    return: scalar loss
    """
    C = _sample_hw(c_feat, num_samples)  # (S, 64)
    S = _sample_hw(s_feat, num_samples)  # (S, 64)

    C = _standardize(C)
    S = _standardize(S)

    # cross-covariance
    cov = (C.t() @ S) / C.size(0)   # (64, 64)

    # Frobenius norm squared (normalized)
    loss = (cov ** 2).mean()
    return loss

def rgb_to_ycbcr(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor (b, 3, h, w) to YCbCr tensor (b, 3, h, w).
    RGB should be in range [0, 1].
    """
    """
        Convert normalized RGB ([0, 1]) to normalized YCbCr ([0, 1]).
        Input shape: (b, 3, h, w)
        Output shape: (b, 3, h, w)
        """
    r, g, b = torch.chunk(rgb_tensor, 3, dim=1)

    # Compute Y, Cb, Cr in normalized form
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128 / 255.0
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128 / 255.0

    ycbcr = torch.cat([y, cb, cr], dim=1)

    # Clamp to [0, 1] to avoid numerical errors
    ycbcr = torch.clamp(ycbcr, 0, 1)
    return ycbcr


def ycbcr_to_rgb(ycbcr_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr tensor (b, 3, h, w) to RGB tensor (b, 3, h, w).
    YCbCr should be in range [0, 1].
    """
    y, cb, cr = torch.chunk(ycbcr_tensor, 3, dim=1)

    # Remove the offset from Cb and Cr components
    cb = cb - 128 / 255.0
    cr = cr - 128 / 255.0
    # Compute R, G, B components
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    rgb = torch.cat([r, g, b], dim=1)
    rgb = torch.clamp(rgb, 0, 1)  # Clamp to handle potential out-of-bound values
    return rgb

def rgb_loss(i1, i2):
    ycbcri1 = rgb_to_ycbcr(i1)
    ycbcri2 = rgb_to_ycbcr(i2)
    cbi1 = ycbcri1[:, 1:2, :, :]
    cri1 = ycbcri1[:, 2:, :, :]
    cbi2 = ycbcri2[:, 1:2, :, :]
    cri2 = ycbcri2[:, 2:, :, :]

    cbloss = F.l1_loss(cbi1, cbi2)
    crloss = F.l1_loss(cri1, cri2)

    loss = crloss + cbloss

    return loss

class Grad_extract(nn.Module):
    def __init__(self):
        super(Grad_extract, self).__init__()
        self.sobelconv=Sobelxy()
    
    def forward(self,x):
        grad = KF.sobel(x)
        grad = torch.clamp(grad, -1e5, 1e5)
        return grad

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        # 添加输入验证
        assert not torch.isnan(image_vis).any(), "NaN values in image_vis"
        assert not torch.isnan(image_ir).any(), "NaN values in image_ir"
        assert not torch.isnan(generate_img).any(), "NaN values in generate_img"

        # 确保值范围合理
        if image_vis.max() > 1e5 or image_ir.max() > 1e5 or generate_img.max() > 1e5:
            print("Warning: Very large values detected in inputs")
        x_in_max=torch.max(image_ir,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=KF.sobel(image_vis)
        ir_grad=KF.sobel(image_ir)
        generate_img_grad=KF.sobel(generate_img)

        y_grad = torch.clamp(y_grad, -1e5, 1e5)
        ir_grad = torch.clamp(ir_grad, -1e5, 1e5)
        generate_img_grad = torch.clamp(generate_img_grad, -1e5, 1e5)

        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def ncc(img1, img2):
    std1 = torch.std(img1, dim=3, keepdim=True)
    std1 = torch.std(std1, dim=2, keepdim=True)
    # std1 = torch.std(std1, dim=1, keepdim=True)
    std2 = torch.std(img2, dim=3, keepdim=True)
    std2 = torch.std(std2, dim=2, keepdim=True)
    # std2 = torch.std(std2, dim=1, keepdim=True)

    mean1 = torch.mean(img1, dim=3, keepdim=True)
    mean1 = torch.mean(mean1, dim=2, keepdim=True)
    mean1 = torch.mean(mean1, dim=1, keepdim=True)
    mean2 = torch.mean(img2, dim=3, keepdim=True)
    mean2 = torch.mean(mean2, dim=2, keepdim=True)
    mean2 = torch.mean(mean2, dim=1, keepdim=True)

    nume = torch.multiply((img1 - mean1), (img2 - mean2))
    nume = torch.mean(nume, dim=3, keepdim=True)
    nume = torch.mean(nume, dim=2, keepdim=True)
    nume = torch.mean(nume, dim=1, keepdim=True)

    deno = torch.multiply(torch.sqrt(std1 + 1e-5), torch.sqrt(std2 + 1e-5))

    ncc1 = nume / (deno + 1e-5)
    ncc1 = torch.clamp(ncc1, -1., 1.)
    ncc = torch.mean(ncc1)

    return ncc

def cc(img1, img2):
    eps = 1e-8
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1) + 1e-8) * torch.sqrt(torch.sum(img2**2, dim=-1) + 1e-8))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


def gradient_loss(img1, img2):
    grad1 = KF.sobel(img1, eps=1e-10)
    grad2 = KF.sobel(img2, eps=1e-10)
    grad_loss = nn.L1Loss()
    loss = grad_loss(grad1, grad2)
    return loss

def grad_fuse_loss(img, th):
    grad = KF.sobel(img)
    ones = torch.ones_like(img)
    grad_mat = torch.where(torch.abs(grad) >= ones, ones, torch.square(torch.abs(grad))/(th**2))
    loss = torch.mean(grad_mat)
    return loss

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

def infonce_loss(img, text1, text2, tau):
    sim_img2text1 = (img @ text1.T).squeeze(0) / tau
    sim_img2text2 = (img @ text2.T).squeeze(0) / tau
    sim_img2text1 = torch.exp(sim_img2text1)
    sim_img2text2 = torch.exp(sim_img2text2)
    Loss_infoce = torch.sum(sim_img2text1) / (torch.sum(sim_img2text1) + torch.sum(sim_img2text2) + 1e-5)
    Loss_infoce = -torch.log(Loss_infoce)

    return Loss_infoce


'''
img1 = torch.randn([24,1,120,120])
img2 = 0.5 * img1

cca = cc(img1, img2)

cci, ccs = ncc(img1, img2)

print(ccs)
'''









