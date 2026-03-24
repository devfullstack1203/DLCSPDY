import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.use_deterministic_algorithms(True)
from ultralytics.nn.newModule.CGLU  import ConvolutionalGLU
from ultralytics.nn.modules import C2PSA, C2f, C3, Conv
from ultralytics.nn.modules.block import PSABlock
__all__ = ['LRSA2','C2PSA_LRSA2','C3k2_LRSA2', 'C3k2_LRLU']
class LRSA2(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pooled_sizes=[11, 8, 6, 4], q_pooled_size=16, q_conv=False):
 
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
 
        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
 
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001
 
        self.norm = nn.LayerNorm(dim)
 
        self.q_pooled_size = q_pooled_size
 
        # Useless code
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None
 
        self.d_convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pooled_sizes])
 
    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        x = x.flatten(2).permute(0, 2, 1)  # B C H W -> B N C
 
        if self.q_pooled_size > 1:
            # Too keep the W/H ratio of the features
            q_pooled_size = (self.q_pooled_size, round(W * float(self.q_pooled_size) / H + self.eps)) \
                if W >= H else (round(H * float(self.q_pooled_size) / W + self.eps), self.q_pooled_size)
 
            # Conduct fixed pooled size pooling on q
            q = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(B, C, H, W), q_pooled_size)
            _, _, H1, W1 = q.shape
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            H1, W1 = H, W
            if self.q_conv is not None:
                x1 = x.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1)
                q = x1 + self.q_conv(x1)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
 
        # Conduct Pyramid Pooling on K, V
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pooled_size, l) in zip(self.pooled_sizes, self.d_convs):
            pooled_size = (pooled_size, round(W * pooled_size / H + self.eps)) if W >= H else (
            round(H * pooled_size / W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x_, pooled_size)
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
 
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))
 
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
 
        # self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)  # B N C
        x = x.transpose(1, 2).reshape(B, -1, C)
 
        x = self.proj(x)
 
        # Bilinear upsampling for residual connection
        x = x.transpose(1, 2).reshape(B, C, H1, W1)
        if self.q_pooled_size > 1:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
class PSABlock_LRSA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = LRSA2(c, num_heads=num_heads)
class C2PSA_LRSA2(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_LRSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
 
class Bottleneck_LRSA(nn.Module):
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
        self.Attention = LRSA2(c2)
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
 
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_LRSA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
 
 
class C3k2_LRSA2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_LRSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        
        
        
class C3k2_LRLU(nn.Module):
    """
    C3k2 mặc định gồm 2 khối:
      1️⃣ LRSA2  - Long Range Spatial Attention
      2️⃣ ConvolutionalGLU - Gated Convolution Block
    Giữ nguyên cấu trúc CSP 2-nhánh tương tự C2f.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,
                 lrsa_heads=4, lrsa_qps=16, attn_drop=0., proj_drop=0., cglu_drop=0.):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # Nhánh đầu: tách 2 nửa (giống C2f)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + 2) * self.c, c2, 1)  # 2 block -> (2+n)*c_

        # Khối 1: LRSA2
        self.lrsa = LRSA2(
            dim=self.c,
            num_heads=lrsa_heads,
            qkv_bias=False,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pooled_sizes=[11, 8, 6, 4],
            q_pooled_size=lrsa_qps,
            q_conv=False
        )

        # Khối 2: ConvolutionalGLU
        self.cglu = ConvolutionalGLU(
            in_features=self.c,
            out_features=self.c,
            drop=cglu_drop
        )

    def forward(self, x):
        """
        Chu trình dữ liệu:
        Input -> Conv1x1 -> Split -> LRSA2 -> CGLU -> Concat -> Conv1x1
        """
        # 1. Giảm chiều, tách thành 2 nhánh
        y1, y2 = self.cv1(x).chunk(2, 1)

        # 2. Nhánh chính: LRSA2 -> CGLU
        y2 = self.lrsa(y2)
        y2 = self.cglu(y2)

        # 3. Nối tất cả: [y1, y2, output các block] = [y1, y2, y2(LRSA), y2(CGLU)]
        y = torch.cat((y1, y2, self.lrsa(y2), self.cglu(y2)), 1)

        # 4. Hợp nhất đầu ra
        return self.cv2(y)
        