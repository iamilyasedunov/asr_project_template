from torch import nn
from hw_asr.base import BaseModel
from .utils import *


class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=11, stride=1, n_groups=1, *args, **kwargs):
        # super().__init__(self, c_in, c_out)
        super().__init__()
        self.base_block = nn.Sequential(
            # k-sized depthwise conv layer with c_out channels
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, stride=stride, groups=n_groups),
            # pointwise convolution
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride),
            # normalization layer
            nn.BatchNorm1d(c_out),
            # ReLU
            nn.ReLU(),
        )

    def forward(self, x):
        return self.base_block(x)


class MainBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=11, stride=1, n_groups=1, *args, **kwargs):
        super().__init__()
        self.base_block = BaseBlock()

    def forward(self, x):
        return self.base_block(x)


class ConvModel(BaseModel):
    def __init__(self, n_feats, model_name, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.model_config = get_model_config(model_name)
        self.n_feats = n_feats
        self.n_class = n_class
        self.model = self.get_model()

    def get_model(self):
        model_list = []
        # for name_block, params in self.model_config.items():
        #     block =

    def forward(self, spectrogram, *args, **kwargs):
        # print(spectrogram.shape)
        logits = self.base_block(spectrogram.transpose(1, 2))
        # print(f"logits shape: {logits.shape}")
        # print(f"calc_shape: {self.calculate_output_length(737, 11)}")
        return {"logits": logits.transpose(1, 2)}

    def transform_input_lengths(self, input_lengths):
        # print(f"input_lengths: {self.calculate_output_length(input_lengths)}")
        return self.calculate_output_length(input_lengths)  # we don't reduce time dimension here
