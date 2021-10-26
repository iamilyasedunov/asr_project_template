import torch
from torch import Tensor
from torch import distributions

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = distributions.Bernoulli(p)
        self._aug = distributions.Normal(*args, **kwargs)

    def __call__(self, data: Tensor):
        if self.p.sample(sample_shape=(1,)).to(torch.bool):
            x = data.unsqueeze(1)
            x = x + self._aug.sample(x.size())
            return x.squeeze(1)
        else:
            return data
