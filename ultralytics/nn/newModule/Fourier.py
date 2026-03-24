import torch
import torch.nn as nn
import numpy as np
from ultralytics.nn.modules import Bottleneck, Conv, C3k2
from ultralytics.nn.modules.block import C3k
__all__ = ['FourierConv','C3k2_FourierConv']
def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)
 
    rng = np.random.RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)
 
class FourierConv(nn.Module):
    def __init__(self, input_chs:int, output_chs:int, HW:int, stride=1, init='he'):
        super(FourierConv, self).__init__()
        num_rows = HW
        num_cols = HW
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols//2 + 1)))
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols//2 + 1)))
        complexinit(self.weights_real, self.weights_imag, init)
        self.size = (num_rows, num_cols)
        self.stride = stride
        self.conv = Conv(input_chs, output_chs, 1 if stride == 1 else 3, s=stride)
    def forward(self, x):
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        x_real, x_imag = x.real, x.imag
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm=None)
        x = self.conv(x)
        return x
 
 
class Bottleneck_FourierConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), HW=None, e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv1 = FourierConv(c1, c_, HW) # 可以去做消融实验，使用一种，还是都使用。
        self.cv2 = FourierConv(c_, c2, HW)
 
 
class C3k_FourierConv(C3k):
    def __init__(self, c1, c2, n=1, HW=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(Bottleneck_FourierConv(c_, c_, shortcut, g, k=(k, k), HW=HW, e=1.0) for _ in range(n)))
 
 
class C3k2_FourierConv(C3k2):
    def __init__(self, c1, c2, n=1, HW=20, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_FourierConv(self.c, self.c, 2, HW, shortcut, g) if c3k else Bottleneck_FourierConv(self.c, self.c,shortcut, g,HW=HW)
            for _ in range(n))
 