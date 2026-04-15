import math
import copy
import torch
import torch.nn.functional as F

from random import random
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
from beartype import beartype
from functools import partial
from collections import namedtuple
from einops.layers.torch import Rearrange
from model.loss import improved_soft_dice_cldice


ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
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


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t):
    return t


def un_normalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def func(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def up_sample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def down_sample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


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


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


def feed_forward_att(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1)
    )


class MIDAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        self.pre_norm_x = LayerNorm(dim)
        self.pre_norm_c = LayerNorm(dim)

        self.to_qkv_x = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_qkv_c = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, c_x):
        b, c, h, w = x.shape

        x = self.pre_norm_x(x)
        c_x = self.pre_norm_c(c_x)

        qkv_x = self.to_qkv_x(x).chunk(3, dim=1)
        qkv_c = self.to_qkv_c(c_x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)
        q_c, _, _ = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_c)

        q_c = q_c * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q_c, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class ViTAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class ViTTransformer(nn.Module):
    def __init__(self, dim, seq_len, depth=4, heads=8, dim_head=64):
        super().__init__()
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
                    nn.Linear(dim * 4, dim)
                )
            ]))

    def forward(self, x):
        x = x + self.pos_emb
        for ln1, attn, ln2, ff in self.layers:
            x = ln1(x)
            x = attn(x) + x
            x = ln2(x)
            x = ff(x) + x
        return x


class ViTModule(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8, height=64, width=64):
        super().__init__()
        seq_len = height * width
        self.transformer = ViTTransformer(
            dim=dim,
            seq_len=seq_len,
            depth=depth,
            heads=heads
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert h * w == self.transformer.pos_emb.shape[1], \
            f"Expected seq_len={self.transformer.pos_emb.shape[1]}, got H*W={h*w}"

        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.transformer(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x


class MIDTransformer(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads)),
                Residual(feed_forward_att(dim)),
                Residual(feed_forward_att(dim)),
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads))
            ]))

    def forward(self, x, c):
        for attn_1, ff_linear_1, ff_linear_2, attn_2 in self.layers:
            x = attn_1(x, c)
            x1 = ff_linear_1(x)
            x2 = ff_linear_2(x)
            x = attn_2(x1, x2)
        return x


@beartype
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        mask_channels=1,
        input_img_channels=3,
        init_dim=None,
        dim_mult: tuple = (1, 2, 4, 8),
        vit_depth=4,
        vit_heads=8,
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head=32,
        attn_heads=4,
        mid_transformer_depth=1,
        self_condition=False,
        resnet_block_groups=4,
        conditioning_klass=None,
        skip_connect_condition_fmap=True
    ):
        super().__init__()

        self.image_size = image_size
        self.mask_channels = mask_channels
        self.self_condition = self_condition
        self.input_img_channels = input_img_channels

        output_channels = mask_channels
        mask_channels = input_img_channels

        init_dim = default(init_dim, dim)

        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding=3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads)

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        self.downs = nn.ModuleList([])
        self.skip_connect_condition_fmap = skip_connect_condition_fmap

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(CBAMLayer(dim_in)),
                down_sample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        final_dim = dim * dim_mult[-1]
        self.vit = ViTModule(dim=final_dim, depth=vit_depth, heads=vit_heads)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = MIDTransformer(mid_dim, depth=mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmap else 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                Residual(CBAMLayer(dim_out)),
                up_sample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(self, x, time, cond, x_self_cond=None):
        skip_connect_c = self.skip_connect_condition_fmap

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        c = self.cond_init_conv(cond)

        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for (block1, block2, attn, d_sample), (cond_block1, cond_block2, cond_attn, cond_d_sample) in zip(self.downs, self.cond_downs):
            x = block1(x, t)
            c = cond_block1(c, t)
            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)

            x = attn(x)
            c = cond_attn(c)

            h.append([x, c] if skip_connect_c else [x])

            x = d_sample(x)
            c = cond_d_sample(c)

        x = self.vit(x)
        c = self.vit(c)

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        x = x + c
        x = self.mid_transformer(x, c)
        x = self.mid_block2(x, t)

        for block1, block2, attn, up_s in self.ups:
            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)

            x = attn(x)
            x = up_s(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(time_steps):
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float64)


def cosine_beta_schedule(time_steps, s=0.008):
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps, dtype=torch.float64)
    alphas_cum_prod = torch.cos(((x / time_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cum_prod = alphas_cum_prod / alphas_cum_prod[0]
    betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])
    return torch.clip(betas, 0, 0.999)


class MedSegDiff(nn.Module):
    def __init__(
        self,
        model,
        time_steps=1000,
        sampling_time_steps=None,
        objective='predict_x0',
        beta_schedule='cosine',
        ddim_sampling_eta=1.
    ):
        super().__init__()

        self.model = model
        self.objective = objective
        self.image_size = model.image_size
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.input_img_channels = self.model.input_img_channels
        self.loss = improved_soft_dice_cldice()

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.)

        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)
        self.sampling_time_steps = default(sampling_time_steps, time_steps)
        assert self.sampling_time_steps <= time_steps

        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cum_prod', alphas_cum_prod)
        register_buffer('alphas_cum_prod_prev', alphas_cum_prod_prev)
        register_buffer('sqrt_alphas_cum_prod', torch.sqrt(alphas_cum_prod))
        register_buffer('sqrt_one_minus_alphas_cum_prod', torch.sqrt(1. - alphas_cum_prod))
        register_buffer('log_one_minus_alphas_cum_prod', torch.log(1. - alphas_cum_prod))
        register_buffer('sqrt_recip_alphas_cum_prod', torch.sqrt(1. / alphas_cum_prod))
        register_buffer('sqrt_recip_m1_alphas_cum_prod', torch.sqrt(1. / alphas_cum_prod - 1))

        posterior_variance = betas * (1. - alphas_cum_prod_prev) / (1. - alphas_cum_prod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_cof_1', betas * torch.sqrt(alphas_cum_prod_prev) / (1. - alphas_cum_prod))
        register_buffer(
            'posterior_mean_cof_2',
            (1. - alphas_cum_prod_prev) * torch.sqrt(alphas) / (1. - alphas_cum_prod)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cum_prod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cum_prod, t, x_start.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cum_prod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recip_m1_alphas_cum_prod, t, x_t.shape)
        )

    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, c, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            predict_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(predict_noise, x_start)

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, clip_de_noised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        predicts = self.model_predictions(x, batched_times, c, x_self_cond)

        x_start = predicts.predict_x_start
        if clip_de_noised:
            x_start.clamp_(-1., 1.)
        return x_start

    @torch.no_grad()
    def p_sample_loop(self, cond):
        x_start = None
        img = cond

        for t in tqdm(reversed(range(0, self.num_time_steps)), desc='sampling time step', total=self.num_time_steps):
            self_cond = x_start if self.self_condition else None
            x_hat_0 = self.p_sample(img, t, cond, self_cond)

            batched_times = torch.full((img.shape[0],), t, device=img.device, dtype=torch.long)
            img_xt = self.q_sample(x_start=x_hat_0, t=batched_times, noise=cond)

            img_xt_sub = img_xt
            if t - 1 != -1:
                batched_times = torch.full((img.shape[0],), t - 1, device=img.device, dtype=torch.long)
                img_xt_sub = self.q_sample(x_start=x_hat_0, t=batched_times, noise=cond)

            img = img - img_xt + img_xt_sub

        img.clamp_(-1, 1)
        img = un_normalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample_loop_ones(self, cond):
        x_start = None
        img = cond
        self_cond = x_start if self.self_condition else None
        t = self.num_time_steps
        img = self.p_sample(img, t - 1, cond, self_cond)
        img = un_normalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, cond_img):
        cond_img = cond_img.to(self.device)
        x0_pred = self.p_sample_loop(cond_img)
        return x0_pred

    @torch.no_grad()
    def sample_ones(self, cond_img):
        cond_img = cond_img.to(self.device)
        return self.p_sample_loop_ones(cond_img)

    def p_losses(self, x_start, t, cond):
        x = self.q_sample(x_start=x_start, t=t, noise=cond)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond).predict_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, cond, x_self_cond)
        model_out = torch.sigmoid(model_out)

        if self.objective == 'predict_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return self.loss(target, model_out)

    def forward(self, img, cond_img, epoch, epochs):
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')
        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)

        b, c, h, w = img.shape
        img_size = self.image_size
        img_channels, mask_channels = self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        times = torch.randint(0, self.num_time_steps, (b,), device=device).long()
        return self.p_losses(img, times, cond_img)


if __name__ == '__main__':
    model = Unet(
        dim=64,
        image_size=512,
        dim_mult=(1, 2, 4, 8),
        mask_channels=1,
        input_img_channels=1,
        resnet_block_groups=2,
        self_condition=False
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")