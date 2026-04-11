# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # 确保参数在正确的设备上
        device = next(self.parameters(), torch.tensor(0)).device
        if not hasattr(self, 'affine_weight') or self.affine_weight.device != device:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features, device=device))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features, device=device))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # 确保所有参数在同一设备上
        if self.affine:
            weight = self.affine_weight.to(x.device)
            bias = self.affine_bias.to(x.device)
            x = x - bias
            x = x / (weight + self.eps*self.eps)
        stdev = self.stdev.to(x.device)
        mean = self.mean.to(x.device)
        x = x * stdev
        if self.subtract_last:
            last = self.last.to(x.device)
            x = x + last
        else:
            x = x + mean
        return x
