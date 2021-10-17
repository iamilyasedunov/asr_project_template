from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        import re
        chars = [self.ind2char[i] for i in inds]
        chars_with_empty_tok = re.sub(r'(\D)\1+', r'\1', "".join(chars))
        text = re.sub(r'\^', r'', chars_with_empty_tok)
        return text

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
