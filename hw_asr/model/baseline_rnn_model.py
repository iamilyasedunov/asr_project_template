from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class BaselineRNNModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, n_layers=3,  *args, **kwargs):
        super().__init__(n_feats, hidden_size, n_layers, n_class, *args, **kwargs)
        self.lstm = nn.LSTM(n_feats, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        # print(f"input shape: {spectrogram.shape}")
        x, hs = self.lstm(spectrogram)
        # print(f"hs shape: {hs}")
        x = self.linear(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
