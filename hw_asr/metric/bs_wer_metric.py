from typing import List

from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        wers = []
        normalized_text = [self.text_encoder.normalize_text(sentence) for sentence in text]
        if hasattr(self.text_encoder, "ctc_beam_search"):
            bs_texts = self.text_encoder.ctc_beam_search(log_probs.cpu())
        else:
            print(f"Error: ctc_beam_search not founded")
            raise NotImplementedError

        for pred_text, target_text in zip(bs_texts, normalized_text):
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
