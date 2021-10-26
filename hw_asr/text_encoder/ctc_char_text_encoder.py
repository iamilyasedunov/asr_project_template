from typing import List, Tuple
from multiprocessing import Pool
import torch
import kenlm

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str],
                 path_to_vocab=None,
                 kenlm_model_path=None):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        unigram_list = None
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if path_to_vocab is not None:
            with open(path_to_vocab) as f:
                unigram_list = [t.lower() for t in f.read().strip().split("\n")]
        # kenlm_model_path = "other/test_clean_normalized.arpa" # lowercase_3-gram.pruned.1e-7.arpa
        self.bs_ctc_decoder = build_ctcdecoder([''] + alphabet,
                                               kenlm_model_path,
                                               unigram_list)

    def ctc_decode(self, inds: List[int]) -> str:
        import re
        chars = [self.ind2char[i] for i in inds]
        chars_with_empty_tok = re.sub(r'(\D)\1+', r'\1', "".join(chars))
        text = re.sub(r'\^', r'', chars_with_empty_tok)
        return text

    def ctc_beam_search(self, logits: torch.tensor) -> List[str]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """

        assert len(logits.shape) == 3
        _, char_length, voc_size = logits.shape
        assert voc_size == len(self.ind2char)

        with Pool(processes=20) as pool:
            texts = self.bs_ctc_decoder.decode_batch(pool=pool, logits_list=logits.detach().numpy())

        return texts
