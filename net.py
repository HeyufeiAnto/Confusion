import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
import numbers
from model import longclip
import numpy
# Resnet
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)
model_clip.eval()
logit_scale = model_clip.logit_scale.exp()

class Branch_Encoder_Ec(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Branch_Encoder_Ec, self).__init__(*args, **kwargs)
        self.enc = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        
    def forward(self, x):
        features = self.enc(x)
        return features

class Branch_Encoder_Exi(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Branch_Encoder_Exi, self).__init__(*args, **kwargs)
        self.enc = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        
    def forward(self, x):
        features = self.enc(x)
        return features

class Branch_Encoder_Exv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Branch_Encoder_Exv, self).__init__(*args, **kwargs)
        self.enc = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        
    def forward(self, x):
        features = self.enc(x)
        return features

class Text_project(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Text_project, self).__init__(*args, **kwargs)
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )
    def forward(self, features, mask):
        features_dot = torch.sum(features, dim=(2,3))
        mask_dot = torch.sum(mask, dim=(2,3))
        features_dot = features_dot / mask_dot
        features_proj = self.proj(features_dot)
        return features_proj
        

class MEB(nn.Module):
    def __init__(self) -> None:
        super(MEB, self).__init__()
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(1, 32, stride=1, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 128, stride=1, padding=1, kernel_size=3),
            nn.ReLU()
        )
    def forward(self, mask, x):
        maskf = self.embedding_layer(mask)
        maskw, maskb = maskf.chunk(2, dim=1)
        output = maskw * x + maskb
        return output

class SF_Fuse(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SF_Fuse, self).__init__(*args, **kwargs)
        self.modu_IR = MEB()
        self.modu_VI = MEB()
        self.project_layer = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fuenc = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
    def forward(self, f1, f2, m1, m2):
        f1_m = self.modu_IR(m1, f1)
        f2_m = self.modu_VI(m2, f2)
        f_fu_m = self.project_layer(torch.cat((f1_m, f2_m), dim=1))
        f_fu = self.fuenc(f_fu_m)
        return f_fu

class Prompt_Project(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Prompt_Project, self).__init__(*args, **kwargs)
        self.project = nn.Linear(512, 128)
        self.Relu = nn.ReLU()
    def forward(self, x):
        x = x.to(self.project.weight.dtype)
        x = self.project(x)
        x = self.Relu(x)
        return x

class CF_Fuse(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CF_Fuse, self).__init__(*args, **kwargs)
        self.en = CF_Enhance()
        self.crf = CRF(64, 8, bias=False)
        self.enc = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.proj = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
            )
    def forward(self, x, y, prompt):
        xc, yc = self.crf(x, y)
        fc = self.enc(self.proj(torch.cat((xc, yc), dim=1)))
        fc_en = self.en(fc, prompt)
        
        return fc_en


class CF_Enhance(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CF_Enhance, self).__init__(*args, **kwargs)
        self.pp = Prompt_Project()
        self.kvnorm = nn.LayerNorm(64)
        self.fnorm = nn.LayerNorm(64)
        self.temperature = nn.Parameter(torch.ones(1, 1))
        
    def forward(self, f, prompt):
        batch_size, seq_len, feat_dim = prompt.shape  
        b, c, h, w = f.shape
        prompt_reshaped = prompt.reshape(-1, feat_dim)
        processed_prompt = self.pp(prompt_reshaped)
        new_feat_dim = processed_prompt.shape[-1]
        processed_prompt = processed_prompt.view(batch_size, seq_len, new_feat_dim)
        pk, pv = processed_prompt.chunk(2, dim=-1)
        pk = self.kvnorm(pk)
        pv = self.kvnorm(pv)
        fq = f.view(b, c, h*w)
        fq = fq.transpose(-2, -1)
        fq = self.fnorm(fq)
        attn = (fq @ pk.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        enhancef = (attn @ pv) 
        enhancef = enhancef.transpose(-2, -1)
        enhancef= enhancef.view(b, c, h, w)
        f_enhance = enhancef + f
        
        return f_enhance


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
# 浅层特征提取

class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[1, 1],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

    def forward(self, img):
        level1_feature = self.patch_embed(img)
        level1_feature = self.encoder_level1(level1_feature)
        return level1_feature
# 共享特征提取

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CRF(nn.Module):
    def __init__(self, dim, num_heads, bias, norm_groups=16):
        super(CRF, self).__init__()
        self.num_heads = num_heads
        self.temperaturex = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperaturey = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvx = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconvx = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_outx = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkvy = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconvy = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.norm1 = nn.GroupNorm(norm_groups, dim)

    def forward(self, cx, cy):
        b, c, h, w = cx.shape

        cxx = self.norm1(cx)
        xqkv = self.qkv_dwconvx(self.qkvx(cxx))
        qx, kx, vx = xqkv.chunk(3, dim=1)

        cyy = self.norm1(cy)
        yqkv = self.qkv_dwconvy(self.qkvy(cyy))
        qy, ky, vy = yqkv.chunk(3, dim=1)

        qx = rearrange(qx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        qy = rearrange(qy, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        ky = rearrange(ky, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        vy = rearrange(vy, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)

        attnyx = (qy @ kx.transpose(-2, -1)) * self.temperaturex
        attnyx = attnyx.softmax(dim=-1)

        outyx = (attnyx @ vx)

        outyx = rearrange(outyx, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        attnxy = (qx @ ky.transpose(-2, -1)) * self.temperaturey
        attnxy = attnxy.softmax(dim=-1)

        outxy = (attnxy @ vy)

        outxy = rearrange(outxy, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        outyx = outyx + cx
        outxy = outxy + cy
        return outyx, outxy

class SEA(nn.Module):
    def __init__(self, dim, num_heads, bias, norm_groups=16):
        super(SEA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, cx):
        b, c, h, w = cx.shape

        cxx = self.norm1(cx)
        xqkv = self.qkv_dwconv(self.qkv(cxx))
        qx, kx, vx = xqkv.chunk(3, dim=1)


        qx = rearrange(qx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        attnx = (qx @ kx.transpose(-2, -1)) * self.temperature
        attnx = attnx.softmax(dim=-1)

        outx = (attnx @ vx)

        outx = rearrange(outx, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)
        outx = self.proj(outx)
        
        return outx

# Shared Information Encoder
class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# Fuse Block
class Fuse_Block(nn.Module):
    def __init__(self): 
        super(Fuse_Block, self).__init__()
        self.project_layer = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.sea = SEA(dim=64, num_heads=8, bias=False)
    def forward(self, x, y):
        fusexy = self.sea(self.project_layer(torch.cat((x, y), dim=1)))
        return fusexy

# Encoder
class Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder, self).__init__()
        self.fuseconv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self,  x_1, x_2):
        x = self.fuseconv(torch.cat((x_1, x_2), dim=1))
        out_enc_level0 = self.encoder_level1(x)
        out_enc_level1 = self.output(out_enc_level0)
        out_put =  self.sigmoid(out_enc_level1)

        return out_put

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

