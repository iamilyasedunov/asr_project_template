import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PeakNormalization(AugmentationBase):
    def __init__(self, sample_rate=16000, *args, **kwargs):
        self.sr = sample_rate
        self._aug = torch_audiomentations.PeakNormalization(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x, self.sr).squeeze(1)
