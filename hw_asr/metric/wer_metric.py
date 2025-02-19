from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        normalized_text = [self.text_encoder.normalize_text(sentence) for sentence in text]
        for log_prob_vec, target_text in zip(predictions, normalized_text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec.numpy())
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
