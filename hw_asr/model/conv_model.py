import torch
from torch import nn
from hw_asr.base import BaseModel
from .utils import *


class BaseBlock(nn.Module):
    def __init__(self, b_block_params, c_in=256, dropout_rate=0.25, *args, **kwargs):
        super().__init__()
        dilation = 1
        num_repeat = b_block_params["R"]
        b_block = []
        in_channels = c_in
        out_channels = b_block_params["C"]
        self.dropout = nn.Dropout(p=dropout_rate)
        padding = dilation * (b_block_params["K"] - 1) // 2
        for i in range(num_repeat):
            b_block.append(nn.Sequential(
                # k-sized depthwise conv layer with c_out channels
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=b_block_params["K"],
                          groups=in_channels, padding=padding),
                # pointwise convolution
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                # normalization layer
                nn.BatchNorm1d(out_channels),
                )
            )
            if i != num_repeat - 1:
                b_block.append(nn.ReLU())
                b_block.append(self.dropout)
            in_channels = out_channels

        self.residual_connection = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        self.base_blocks = nn.Sequential(*b_block)

    def forward(self, x):
        output = self.base_blocks(x)
        output += self.residual_connection(x)
        output = nn.ReLU()(output)
        output = self.dropout(output)
        return output


class QuartzNet(BaseModel):
    def __init__(self, model_name, n_feats, n_class, dropout_rate=0.2, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.model_config = get_model_config(model_name, n_class)
        self.n_feats = n_feats
        self.n_class = n_class
        self.c1_block, self.c2_block, self.c3_block, self.c4_block = None, None, None, None
        self.b_blocks = []
        self.b_blocks_residuals = []
        self.dropout_rate = dropout_rate
        self.get_model()

    def get_model(self):
        c1_block_params = self.model_config["C1"]
        c2_block_params = self.model_config["C2"]
        c3_block_params = self.model_config["C3"]
        c4_block_params = self.model_config["C4"]
        с1_block_padding = (c1_block_params["K"] - 1) // 2
        self.c1_block = nn.Sequential(
            nn.Conv1d(self.n_feats, self.n_feats, kernel_size=c1_block_params["K"],
                      stride=2, padding=с1_block_padding, groups=self.n_feats),
            # pointwise convolution
            nn.Conv1d(self.n_feats, c1_block_params["C"], kernel_size=1),
            # normalization layer
            nn.BatchNorm1d(c1_block_params["C"]),
            # ReLU
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        # len = calculate_output_length(self.n_feats, c1_block_params["K"], stride=2, padding=с1_block_padding)
        in_channels = c1_block_params["C"]
        for i in range(1, 6):
            block_config = self.model_config["B" + str(i)]
            self.b_blocks.append(BaseBlock(block_config, in_channels, self.dropout_rate))
            self.b_blocks_residuals.append(nn.Conv1d(in_channels, block_config["C"], kernel_size=1))
            in_channels = block_config["C"]
        self.b_blocks = nn.ModuleList(self.b_blocks)
        self.b_blocks_residuals = nn.ModuleList(self.b_blocks_residuals)
        self.c2_block = nn.Sequential(
            nn.Conv1d(self.model_config["B5"]["C"], c2_block_params["C"], kernel_size=c2_block_params["K"],
                      dilation=2, padding=c2_block_params["K"] - 1, groups=self.model_config["B5"]["C"]),
            # pointwise convolution
            nn.Conv1d(c2_block_params["C"], c2_block_params["C"], kernel_size=1),
            # normalization layer
            nn.BatchNorm1d(c2_block_params["C"]),
            # ReLU
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.c3_block = nn.Sequential(
            # pointwise convolution
            nn.Conv1d(c2_block_params["C"], c3_block_params["C"], kernel_size=c3_block_params["K"]),
            # normalization layer
            nn.BatchNorm1d(c3_block_params["C"]),
            # ReLU
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.c4_block = nn.Sequential(
            nn.Conv1d(c3_block_params["C"], c4_block_params["C"], kernel_size=c4_block_params["K"]),
        )

    def forward(self, spectrogram, *args, **kwargs):
        #print(f"spectrogram: {spectrogram.transpose(1, 2).shape}")
        outputs = self.c1_block(spectrogram.transpose(1, 2))
        #print(f"c1_block: {outputs.shape}")
        for block, residual in zip(self.b_blocks, self.b_blocks_residuals):
            outputs = block(outputs) + residual(outputs)
        #print(f"b_blocks: {outputs.shape}")
        outputs = self.c2_block(outputs)
        #print(f"c2_block: {outputs.shape}")
        outputs = self.c3_block(outputs)
        #print(f"c3_block: {outputs.shape}")
        outputs = self.c4_block(outputs)
        #print(f"c4_block: {outputs.shape}")


        #logits = self.base_block(spectrogram.transpose(1, 2))
        # print(f"logits shape: {logits.shape}")
        # print(f"calc_shape: {self.calculate_output_length(737, 11)}")
        #return {"logits": logits.transpose(1, 2)}
        return {"logits": outputs.transpose(1, 2)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
