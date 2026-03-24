import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt
from ultralytics.nn.modules import C3, C2f
__all__ =['MSCA','GCBAM','C3k2_GCBAM']
 
#第一即插即用模块是 MSCA模块
class MSCA(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSCA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
 
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2
 
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.attn3 = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)
 
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])
 
        self.layer_norm = nn.LayerNorm(dim)
 
        self.topk = topk  # False True
 
    def forward(self,data) :
        # x0 = x
        x, y=data
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1)
 
        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C = y.shape
        # print(y.shape)
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
 
        attn = (q @ k.transpose(-2, -1)) * self.scale
 
        # print(self.k1,self.k2)
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)
 
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)
 
        out = out1 * self.attn1 + out2 * self.attn2  # + out3 * self.attn3
        # out = out1 * self.attn1 + out2 * self.attn2
 
        x = out.transpose(1, 2).reshape(B, N, C)
 
        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)
        # x = x + x0
        return x
 
#第二即插即用模块是 GCBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        x = x * attention
        return x
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)
 
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
class GCBAM(nn.Module):
    def __init__(self, channel, group=8, cov1=1, cov2=1):
        super().__init__()
        self.cov1 = None
        self.cov2 = None
        if cov1 != 0:
            self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.group = group
        cbam = []
        for i in range(self.group):
            cbam_ = CBAM(channel // group)
            cbam.append(cbam_)
 
        self.cbam = nn.ModuleList(cbam)
        self.sigomid = nn.Sigmoid()
        if cov2 != 0:
            self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)
 
    def forward(self, x):
        x0 = x
        if self.cov1 != None:
            x = self.cov1(x)
        y = torch.split(x, x.size(1) // self.group, dim=1)
        mask = []
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)
            y_ = self.sigomid(y_)
 
            mean = torch.mean(y_, [1, 2, 3])
            mean = mean.view(-1, 1, 1, 1)
 
            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)
            mask.append(mk)
 
        mask = torch.cat(mask, dim=1)
        # print(mask.shape)
        x = x * mask
        if self.cov2 != None:
            x = self.cov2(x)
        x = x + x0
        return x
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class Bottleneck_GCBAM(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = GCBAM(c2)
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
 
class C3k_GCBAM(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_GCBAM(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
 
class C3k2_GCBAM(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_GCBAM(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_GCBAM(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
 