from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, n_groups=1, *args, **kwargs):
        super(BaseBlock).__init__()
        self.base_block = nn.Sequential(
            # k-sized depthwise conv layer with c_out channels
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, groups=n_groups),
            # pointwise convolution
            nn.Conv1d(c_in, c_out, kernel_size=1),
            # normalization layer
            nn.BatchNorm1d(c_out),
            # ReLU
            nn.ReLU(),
        )

    def forward(self, x):
        return self.base_block(x)

class ConvModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = BaseBlock()

    def forward(self, spectrogram, *args, **kwargs):
        print(spectrogram.shape)
        return self.net(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
