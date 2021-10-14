from torch import nn
from torch.nn import Sequential
from .utils import calculate_output_length
from hw_asr.base import BaseModel


class BaselineRNNModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, kernel_size=33,
                 stride=1, n_groups=1, n_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, hidden_size=512, kernel_size=33,
                         stride=1, n_groups=1, n_layers=3, *args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

        conv_hidden_size = int(hidden_size / 2)
        self.base_block = nn.Sequential(
            # k-sized depthwise conv layer with c_out channels
            nn.Conv1d(n_feats, n_feats, kernel_size=kernel_size, stride=stride, groups=n_groups),
            # pointwise convolution
            nn.Conv1d(n_feats, conv_hidden_size, kernel_size=1, stride=stride),
            # normalization layer
            nn.BatchNorm1d(conv_hidden_size),
            # ReLU
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_hidden_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        # print(f"spectrogram: {spectrogram.shape}")
        x = self.base_block(spectrogram.transpose(1, 2))
        # print(f"conv: {x.shape}")
        x, _ = self.lstm(x.transpose(1, 2))
        # print(f"lstm: {x.shape}")
        x = self.linear(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return calculate_output_length(input_lengths, self.kernel_size, self.stride)