
import math
import copy

import numpy as np
import torch
import torch.nn.functional as F

from random import random
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
from beartype import beartype
from functools import partial
from torch.fft import fft2, ifft2
from collections import namedtuple
from einops.layers.torch import Rearrange
# from typing import ForwardRef as ForwardRef
from ColdSeg.loss import improved_soft_dice_cldice
from ColdSeg.cbdice_loss import SoftcbDiceLoss
# constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


class SELayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


    



# 判断变量是否存在
def exists(x):
    return x is not None


# 变量的默认选择
def default(val, d):
    # 判断变量是否存在, 如果存在, 直接返回结果. 否则进行变量2的判断
    if exists(val):
        return val

    # 判断输入变量2是否为函数, 如果为函数, 返回函数结果, 否则直接返回变量2
    if callable(d):
        return d()
    else:
        return d


# 残差链接，直接返回结果
def identity(t):
    return t


# 归一化函数, 从 [0, 1] 放缩至 [-1, 1]
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


# 归一化函数, 从 [-1, 1] 放缩至 [0, 1]
def un_normalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# 创建学习率更新策略
def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def func(x):
        """
        学习率调整函数
        根据 step 数返回一个学习率倍率因子，注意在训练开始之前，pytorch 会提前调用一次 lr_scheduler.step() 方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup 过程中 lr 倍率因子从 warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup 后 lr 倍率因子从 1 -> 0
            # 参考 deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)


# 残差链接模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# 上采样模块
def up_sample(dim, dim_out=None):
    up_s = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), (3, 3), padding=1))
    return up_s


# 下采样模块
def down_sample(dim, dim_out=None):
    down_s = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), # 将通道数乘以 4，高宽减半
        nn.Conv2d(dim * 4, default(dim_out, dim), (1, 1)))
    return down_s


# 层归一化模块
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):  # dim: 输入数据的维度 , bias: 是否添加偏置
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


# 正弦位置编码模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# 建立 卷积-归一化-激活 模块  Conv -> GroupNorm -> SiLU
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.norm = nn.GroupNorm(groups, dim_out) # groups将通道分成若干组，并在每个组内进行归一化。
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# 构建残差模块  (Conv -> GroupNorm -> SiLU)*2 + Residual
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()

        # 建立卷积-归一化-激活模块
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # 建立通道转换卷积层
        self.res_conv = nn.Conv2d(dim, dim_out, (1, 1)) if dim != dim_out else nn.Identity()

        # 时间编码
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

    def forward(self, x, time_emb=None):
        # 获取编码后的结果, 拆分为 scale 和 shift
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):# true
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        # 建立卷积-归一化-激活模块
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        # 通道转换后进行残差连接
        return h + self.res_conv(x)


# 前向传播 通道维度的转换 Conv*2 dim-> dim*4 -> dim
def feed_forward_att(dim, mult=4):
    inner_dim = int(dim * mult)
    feed_forward_linear = nn.Sequential(LayerNorm(dim),
                                        nn.Conv2d(dim, inner_dim, (1, 1)),
                                        nn.GELU(),
                                        nn.Conv2d(inner_dim, dim, (1, 1)))
    return feed_forward_linear


# 线性注意力机制
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads  # 32*4=128

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm = LayerNorm(dim) # 层归一化 对每个通道，将每个元素减去该通道的均值，然后除以该通道的标准差
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False) #  (b, c, h, w)->(b, 128*3, h, w)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, (1, 1)), LayerNorm(dim))

    def forward(self, x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1) # 在 dim=1 ,将数据划分为 Q K V
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 线性注意力机制的计算过程
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 获取线性注意力机制的输出结果
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)

# 多头注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1))

    def forward(self, x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 获取 QKV 的计算结果
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # 维度转换后获取网络输出
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

# conditional cross-attention module
class MIDAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm_x = LayerNorm(dim)
        self.pre_norm_c = LayerNorm(dim)

        self.to_qkv_x = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_qkv_c = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, (1, 1))

    def forward(self, x, c_x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm_x(x)
        c_x = self.pre_norm_c(c_x)

        qkv_x = self.to_qkv_x(x).chunk(3, dim=1)
        qkv_c = self.to_qkv_c(c_x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)
        q_c, k_c, v_c = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_c)

        # 获取 QKV 的计算结果
        q_c = q_c * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q_c, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # 维度转换后获取网络输出
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# 通道注意力机制 CAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, (1, 1), bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


# 空间注意力机制 SAM
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# Contrast Enhancement Module
# FFT -> CEM
class CEMLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # 超参数设置(注意力头数目 归一化因子 注意力隐藏层维度)
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        # 设置归一化层、QKV转换层、注意力输出层
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, (1, 1), bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, (1, 1)), LayerNorm(dim))

        # 空间和通道注意力机制
        self.channel_attention = ChannelAttentionModule(hidden_dim)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        # 获取数据维度
        b, c, h, w = x.shape

        # 数据归一化后划分为 Q K V 注意力机制
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # SAM 和 CAM 获取过程
        qkv = list(qkv)
        qkv[0] = self.channel_attention(qkv[0]) * qkv[0] + qkv[0]
        qkv[1] = self.spatial_attention(qkv[1]) * qkv[1] + qkv[1]

        # 格式转换
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 线性注意力机制的计算过程
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 获取线性注意力机制的输出结果
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)

# ============================ViT=======================
class ViTAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        """
        输入形状: (B, Seq_len, Dim)
        输出形状: (B, Seq_len, Dim)
        """
        B, N, _ = x.shape
        h = self.heads

        # 生成QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, h, -1).permute(0, 2, 1, 3), qkv)

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 特征融合
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class ViTTransformer(nn.Module):
    def __init__(self, dim, seq_len, depth=4, heads=8, dim_head=64):
        super().__init__()
        # 添加可训练的位置嵌入，并用 Gaussian 初始化
        self.pos_emb = nn.Parameter(torch.empty(1, seq_len, dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                ViTAttention(dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 4, dim))
            ]))

    def forward(self, x):#(B,H*W,C)
        x = x + self.pos_emb
        for ln1, attn, ln2, ff in self.layers:
            # 注意力残差连接
            x = ln1(x)
            x = attn(x) + x

            # 前馈残差连接
            x = ln2(x)
            x = ff(x) + x
        return x


class ViTModule(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8, height=64, width=64):
        super().__init__()
        seq_len = height * width  # 64*64=4096
        self.transformer = ViTTransformer(
            dim=dim,
            seq_len=seq_len,
            depth=depth,
            heads=heads
        )

    def forward(self, x):
        """
        输入形状: (B, C, H, W)
        输出形状: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H * W == self.transformer.pos_emb.shape[1], \
            f"Expected seq_len={self.transformer.pos_emb.shape[1]}, got H*W={H*W}"
        # 转换为序列 (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 通过Transformer
        x = self.transformer(x)

        # 恢复空间维度
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x
# ============================ViT=======================
    
# Transformer 网络结构
class Transformer(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, dim_head=dim_head, heads=heads)),
                Residual(feed_forward_att(dim))]))

    def forward(self, x):
        for attn, ff_linear in self.layers:
            x = attn(x)
            x = ff_linear(x)
        return x
# Conditional Attention Transformer 网络结构

# MIDTransformer通过交叉注意力机制，融合主路径和条件路径的特征。
# feed_forward_att *2
# MIDAttention
class MIDTransformer(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads)),
                Residual(feed_forward_att(dim)),
                Residual(feed_forward_att(dim)),
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads))]))

    def forward(self, x, c):
        for attn_1, ff_linear_1, ff_linear_2, attn_2 in self.layers:
            x = attn_1(x, c)
            x1 = ff_linear_1(x)
            x2 = ff_linear_2(x)
            x = attn_2(x1, x2)
        return x


# 扩散模型的FFT编码过程
class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()

        # 初始化调制高频的注意力图
        self.ff_theta = nn.Parameter(torch.ones(dim, 1, 1))
        self.ff_parser_attn_map_r = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))
        self.ff_parser_attn_map_i = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        # 输入变量归一化
        self.norm_input = LayerNorm(dim, bias=True)

        # 构建残差模块
        self.block = ResnetBlock(dim, dim)

        # 自注意力机制
        self.attention_f = CEMLinearAttention(dim, heads=4, dim_head=32)

    def forward(self, x):
        # 调制高频的注意力图
        x_type = x.dtype

        # 二维傅里叶变换
        z = fft2(x)

        # 获取傅里叶变换后的 实部 和 虚部
        z_real = z.real
        z_imag = z.imag

        # 频域滤波器 保持低频，增强高频
        # 可学习高频滤波 或者 高频滤波器 (实部 和 虚部的加权处理)
        z_real = z_real * self.ff_parser_attn_map_r
        z_imag = z_imag * self.ff_parser_attn_map_i

        # 合并为复数形式
        z = torch.complex(z_real * self.ff_theta, z_imag * self.ff_theta)

        # 反变换后只需要实部，虚部 为误差
        z = ifft2(z).real

        # 格式转换
        z = z.type(x_type)

        # 条件变量和输入变量的融合
        norm_z = self.norm_input(z)

        # 利用自注意力机制增强学习到的特征
        norm_z = self.attention_f(norm_z + x)

        # 添加一个额外的块以允许更多信息集成，在条件块之后有一个下采样（但也许有一个比下采样之前更好的条件）
        return self.block(norm_z)


@beartype
class Unet(nn.Module):
    def __init__(self, dim, image_size, mask_channels=1, input_img_channels=3, init_dim=None,
                 dim_mult: tuple = (1, 2, 4, 8), vit_depth=4,vit_heads=8, full_self_attn: tuple = (False, False, False, True), attn_dim_head=32,
                 attn_heads=4, mid_transformer_depth=1, self_condition=False, resnet_block_groups=4,
                 conditioning_klass=Conditioning, skip_connect_condition_fmap=True):
        """
        :param dim: 基础维度
        :param image_size: 图像大小
        :param init_dim: 初始维度 None
        :param dim_mult: 维度乘子 tuple = (1, 2, 4, 8)

        :param attn_dim_head: 注意力机制的基础维度  32
        :param attn_heads: 注意力机制的多头数目 4

        :param input_img_channels: 输入原始图像通道数目 3
        :param mask_channels: 输入掩码通道数目(无自条件时, 输出通道数目) 1
        :param mid_transformer_depth: Transformer 深度 1
        :param full_self_attn: 自注意力机制 tuple = (False, False, False, True)
        :param self_condition: 自条件引导 False
        :param resnet_block_groups: 残差模块的组卷积    8
        :param conditioning_klass: 条件模块  Conditioning扩散模型的FFT编码过程
        :param skip_connect_condition_fmap: 解码部分是否采用编码中的条件模块输出 False 不采用FFT
        """
        super().__init__()

        # 超参数的确定
        self.image_size = image_size
        self.mask_channels = mask_channels # 1
        self.self_condition = self_condition # False
        self.input_img_channels = input_img_channels # 3

        # 更改了输入通道数目
        output_channels = mask_channels# 输出通道数1
        mask_channels = input_img_channels # 输入掩码通道数1

        # 确定初始转换维度
        init_dim = default(init_dim, dim)

#step 1 初始卷积
        # 对输入和条件图像分别进行卷积，统一通道数，提取初始特征.
        self.init_conv = nn.Conv2d(mask_channels, init_dim, (7, 7), padding=3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, (7, 7), padding=3)

        # 不同层的输入和输出维度
        # dim_mult: tuple = (1, 2, 4, 8) 通道倍增因子
        dims = [init_dim, *map(lambda m: dim * m, dim_mult)] # [64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:])) # [(64, 128), (128, 256), (256, 512)]

        # 残差模块（Covn*2+res）
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # 扩散模型需要处理不同时间步的噪声，通过SinusoidalPosEmb生成时间嵌入
        time_dim = dim * 4 # 256
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),    #  正弦位置编码
                                      nn.Linear(dim, time_dim),
                                      nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        # 注意力机制相关参数
        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads) # 32维  4头

        # 获取卷积模块的层数
        num_resolutions = len(in_out)# 4
        assert len(full_self_attn) == num_resolutions

        # 参数初始化和赋值
        curr_fmap_size = image_size
        self.downs = nn.ModuleList([])
        #self.conditioners = nn.ModuleList([])
        self.skip_connect_condition_fmap = skip_connect_condition_fmap

        # 下采样编码模块
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            # 判断是否为最后的卷积模块
            is_last = ind >= (num_resolutions - 1)  # ind >= 3

            # 添加下采样模块
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                # SELayer(dim_in), # 添加SE模块
                Residual(CBAMLayer(dim_in)),
                down_sample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, (3, 3), padding=1)]))

            # 特征图规模减半 下采样
            if not is_last:
                curr_fmap_size //= 2
                
        #===========================================
        # # # 中间层模块 利用 Transformer 代替残差连接
        final_dim = dim * dim_mult[-1]
        self.vit = ViTModule(
            dim=final_dim,
            depth=vit_depth,
            heads=vit_heads
        )
        #============================================
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = MIDTransformer(mid_dim, depth=mid_transformer_depth, **attn_kwargs)  # MIDTransformer通过交叉注意力机制，融合主路径和条件路径的特征。
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 条件编码路径将与主编码路径相同
        self.ups = nn.ModuleList([])
        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # 上采样解码模块
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            # 判断是否为最后的卷积模块
            is_last = ind == (len(in_out) - 1)

            # 是否使用跳跃连接
            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmap else 1)  # False -> 1 -> dim_in*1

            # 添加上采样模块
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                Residual(CBAMLayer(dim_out)),
                up_sample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, (3, 3), padding=1)]))

        # 最后的输出层
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, (1, 1))

    def forward(self, x, time, cond, x_self_cond=None):
        # 是否采用跳跃连接
        skip_connect_c = self.skip_connect_condition_fmap  # False

        # 开始是否将条件合并到输入中
        if self.self_condition: # False
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # 输入变量和条件变量的初始卷积过程
        x = self.init_conv(x) # x: [1, 3, 256, 256] -> x: [1, 64, 256, 256]
        c = self.cond_init_conv(cond)

        # 获取初始卷积后的输出, 用于最后的拼接
        r = x.clone() # r: [1, 64, 256, 256]

        # 时间编码模块
        t = self.time_mlp(time)

        # 下采样编码阶段
        h = []
        for (block1, block2,attn, d_sample), (cond_block1, cond_block2, cond_attn,
                                               cond_d_sample) in zip(self.downs, self.cond_downs):
            # 卷积编码模块 + 条件编码模块
            x = block1(x, t)
            c = cond_block1(c, t)

            # 保存卷积和条件编码结果
            h.append([x, c] if skip_connect_c else [x])

            # 卷积编码模块 + 条件编码模块
            x = block2(x, t)
            c = cond_block2(c, t)

            # 添加SE模块
            #x = SE(x)
            #c = cond_SE(c)

            # 注意力模块和条件注意力模块输出
            x = attn(x)
            c = cond_attn(c)

            # 保存编码结果
            h.append([x, c] if skip_connect_c else [x])

            # 下采样模块 条件下采样模块
            x = d_sample(x)
            c = cond_d_sample(c)
        x = self.vit(x)
        c = self.vit(c)

        # 卷积和条件的中间层编码模块
        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        # 条件编码和编码的融合
        x = x + c

        # 中间层编码的注意力机制
        x = self.mid_transformer(x, c)
        x = self.mid_block2(x, t)

        # 上采样解码模块
        for block1, block2, attn, up_s in self.ups:
            # 跳跃连接
            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)

            # 注意力机制
            x = attn(x)

            # 上采样模块
            x = up_s(x)

        # 合并输出和初始卷积后的输出
        x = torch.cat((x, r), dim=1)

        # 最后的卷积层
        x = self.final_res_block(x, t)
        return self.final_conv(x)



def extract(a, t, x_shape):  # self.sqrt_recip_alphas_cum_prod, t, x_t.shape
    b, *_ = t.shape
    out = a.gather(-1, t) # ：从a中提取索引为t的值。
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # 将提取出的值重新调整形状。b保持不变，其余维度填充为1，以匹配x_shape的形状。


# 线性采样方案
def linear_beta_schedule(time_steps):
    scale = 1000 / time_steps       # 1
    beta_start = scale * 0.0001     # 0.001
    beta_end = scale * 0.02         # 0.02
    return torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float64) # [0.001, 0.0012, 0.0014, ..., 0.02]


# 正弦采样方案
def cosine_beta_schedule(time_steps, s=0.008):
    steps = time_steps + 1    # 1001
    x = torch.linspace(0, time_steps, steps, dtype=torch.float64)  # [0, 1, 2, ..., 1000]
    alphas_cum_prod = torch.cos(((x / time_steps) + s) / (1 + s) * math.pi * 0.5) ** 2 #
    alphas_cum_prod = alphas_cum_prod / alphas_cum_prod[0]  # 归一化
    betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])
    return torch.clip(betas, 0, 0.999) #


# 医学图像分割模型
class MedSegDiff(nn.Module):
    def __init__(self, model, time_steps=1000, sampling_time_steps=None, objective='predict_x0',
                 beta_schedule='cosine', ddim_sampling_eta=1., loss_alpha=0.5, loss_beta=0.5, loss_gamma=1.0):
        """
        :param model: 分割模型 UNet
        :param time_steps: 加噪的步长
        :param sampling_time_steps: 采样步长
        :param objective: 预测目标
        :param beta_schedule: 加噪方案
        :param ddim_sampling_eta: 采样率
        """
        super().__init__()

        # 参数的赋值
        self.model = model
        self.objective = objective
        self.image_size = model.image_size
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.input_img_channels = self.model.input_img_channels
        
        
        self.loss = improved_soft_dice_cldice(alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma)

        # 线性采样、正弦采样, 获取 beta
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 获取 alpha 值和累乘结果
        alphas = 1. - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0) # 沿着指定的维度（dim=0）计算累积乘积
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.) #用于计算前一步的累积乘积， pad 函数就是用于对张量进行填充的

        # 根据加噪长度 获取加噪步长和采样步长
        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)
        self.sampling_time_steps = default(sampling_time_steps, time_steps)
        assert self.sampling_time_steps <= time_steps

        # 默认采样时间步数到训练时的时间步数
        # 如果采样时间步数小于训练时间步数，则使用 ddim 采样方案
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        # 辅助函数，用于将缓冲区从 float64 注册到 float32
        # 函数注册的缓冲区在模型的前向传播过程中不会被视为模型参数，因此不会在反向传播过程中更新。
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))
        # 获取 beta 值和累乘结果
        register_buffer('betas', betas)
        register_buffer('alphas_cum_prod', alphas_cum_prod)
        register_buffer('alphas_cum_prod_prev', alphas_cum_prod_prev)
        # 扩散模型相关公式的计算 q(x_t | x_{t-1})
        register_buffer('sqrt_alphas_cum_prod', torch.sqrt(alphas_cum_prod))
        register_buffer('sqrt_one_minus_alphas_cum_prod', torch.sqrt(1. - alphas_cum_prod))
        register_buffer('log_one_minus_alphas_cum_prod', torch.log(1. - alphas_cum_prod))
        register_buffer('sqrt_recip_alphas_cum_prod', torch.sqrt(1. / alphas_cum_prod))
        register_buffer('sqrt_recip_m1_alphas_cum_prod', torch.sqrt(1. / alphas_cum_prod - 1))

        # 后验计算过程 q(x_{t-1} | x_t, x_0)
        # 后验方差计算
        posterior_variance = betas * (1. - alphas_cum_prod_prev) / (1. - alphas_cum_prod)

        register_buffer('posterior_variance', posterior_variance)

        # 下面: 由于扩散链开始时的后验方差为 0 而剪裁的对数计算
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # 后验均值的相关系数
        register_buffer('posterior_mean_cof_1', betas * torch.sqrt(alphas_cum_prod_prev) / (1. - alphas_cum_prod))
        register_buffer('posterior_mean_cof_2',
                        (1. - alphas_cum_prod_prev) * torch.sqrt(alphas) / (1. - alphas_cum_prod))

    @property
    def device(self):
        return next(self.parameters()).device

    # 前向过程
    def q_sample(self, x_start, t, noise):
        """ 前向扩散过程(重参数化采样), 从 q (x_t | x_0) 中采样， 获得 x_t """
        # x_t = sqrt_alpha*x_0 +  sqrt_one_minus_alphas_cum_prod
        return (extract(self.sqrt_alphas_cum_prod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cum_prod, t, x_start.shape) * noise)

    # 预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        """ 根据真实的 x_t 和模型预测的 x_0, 获取添加的噪声 """
        return ((extract(self.sqrt_recip_alphas_cum_prod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recip_m1_alphas_cum_prod, t, x_t.shape))

    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        """ 模型的预测过程 """
        # 获取 UNet 网络的输出
        model_output = self.model(x, t, c, x_self_cond) # x_t, t, cond_img -> x_0

        # 是否对输出结果进行限制
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        # 直接预测逆扩散结果
        if self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            predict_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(predict_noise, x_start) # 字典存储(predict, x_0_bar)

    # 去噪过程
    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, clip_de_noised=True): # x_T, t, cond_img
        """ 通过神经网络预测均值和方差, 即通过x_t 预测 x_{t - 1} 的均值和方差，也包括 x_0 的预测"""
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        predicts = self.model_predictions(x, batched_times, c, x_self_cond)

        # 获取预测的 x_0
        x_start = predicts.predict_x_start
        if clip_de_noised:
            x_start.clamp_(-1., 1.)
        return x_start

    # 完整逆向采样过程
    @torch.no_grad()
    def p_sample_loop(self, cond):
        """ 推理过程中, 给定 x_t 采样 x_{t-1}, 递归采样获取 x_0, 样本恢复过程 """
        # 设置条件图像, 获取输入噪声
        x_start = None
        img = cond  # cond_img

        # 循环采样过程, 显示加载器
        for t in tqdm(reversed(range(0, self.num_time_steps)), desc='sampling time step', total=self.num_time_steps):
            # 判断是否采用自条件 进行限制
            self_cond = x_start if self.self_condition else None
            # 去噪
            x_hat_0 = self.p_sample(img, t, cond, self_cond)  # x_0_bar
          
            # 图像混合 t 时间
            batched_times = torch.full((img.shape[0],), t, device=img.device, dtype=torch.long)
            # 加噪
            img_xt = self.q_sample(x_start=x_hat_0, t=batched_times, noise=cond) # x_t
            # 图像混合 t - 1 时间
            img_xt_sub = img_xt # x_t
            if t - 1 != -1:
                batched_times = torch.full((img.shape[0],), t - 1, device=img.device, dtype=torch.long)
                img_xt_sub = self.q_sample(x_start=x_hat_0, t=batched_times, noise=cond) # x_{t-1}

            # 图像更新
            img = img - img_xt + img_xt_sub 

        img.clamp_(-1, 1)
        # 反标准化
        img = un_normalize_to_zero_to_one(img)
        return img

    #
    @torch.no_grad()
    def p_sample_loop_ones(self, cond):
        """ 推理过程中, 给定 x_t 采样 x_{t-1}, 递归采样获取 x_0, 样本恢复过程 """
        # 设置条件图像, 获取输入噪声
        x_start = None
        img = cond

        # 判断是否采用自条件 进行限制
        self_cond = x_start if self.self_condition else None

        # 获得 UNet 网络的预测值 更新
        t = self.num_time_steps
        img = self.p_sample(img, t - 1, cond, self_cond)

        # 反标准化
        img = un_normalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, cond_img):
        cond_img = cond_img.to(self.device)
        x0_pred = self.p_sample_loop(cond_img)
        return x0_pred

    @torch.no_grad()
    def sample_ones(self, cond_img):

        # 将条件图像添加至运行设备
        cond_img = cond_img.to(self.device)

        # 判断是否采用加速采样
        sample_fn = self.p_sample_loop_ones

        # 返回预测采样结果 img
        return sample_fn(cond_img)



    def p_losses(self, x_start, t, cond): # self.p_losses(img, times, cond_img)
        """ 损失计算过程 """
        # 根据噪声生成加噪后图像
        x = self.q_sample(x_start=x_start, t=t, noise=cond)

        # 如果加入自条件，50% 的时间，根据 UNet 的当前时间和条件预测 x_start，这种技术将使训练速度减慢 25%，但似乎会显着降低 FID
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond).predict_x_start
                x_self_cond.detach_()

        # 预测
        model_out = self.model(x, t, cond, x_self_cond) # x_t -> x_0
        # 对模型输出应用 Sigmoid，确保值在 [0, 1]
        model_out = torch.sigmoid(model_out)

        # 选择预测目标
        if self.objective == 'predict_x0':
            target = x_start # mask
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        return self.loss( target, model_out)
    

    def forward(self, img, cond_img, epoch, epochs):
        """ 前向计算过程, 直接获取损失 """
        # 数据格式转换
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')
        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        # 获取运行设备
        device = self.device

        # 将输入和条件图像添加到运行设备
        img, cond_img = img.to(device), cond_img.to(device)

        # 对图像的大小进行判断, 并给出警告

        b, c, h, w = img.shape
        img_size = self.image_size
        img_channels, mask_channels = self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        # # 生成时间编码
        times = torch.randint(0, self.num_time_steps, (b,), device=device).long()

        # 对图像进行归一化
        #img = normalize_to_neg_one_to_one(img)

        # 计算损失
        return self.p_losses(img, times, cond_img)






if __name__ == '__main__':
    model = Unet(dim=64, image_size=512,dim_mult=(1, 2, 4, 8), mask_channels=1, input_img_channels=1,resnet_block_groups=2, self_condition=False).cuda()
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

