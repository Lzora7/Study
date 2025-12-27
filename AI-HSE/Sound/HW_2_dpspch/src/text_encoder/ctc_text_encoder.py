import re
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoderсн
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        # TODO
        """
        Decode sequence of token indices to text using CTC rules.
        Removes blanks and merges repeated characters.
        
        Args:
            inds: list of token indices or numpy array of token indices.
        Returns:
            decoded_text: string after CTC decoding.
        """

        # got nothing
        if len(inds) == 0:
            return ""
        
        # convert to list if needed
        if hasattr(inds, 'tolist'):
            inds = inds.tolist()
        inds = [int(ind) for ind in inds]
        
        # remove blanks and merge repeats
        decoded = []
        prev_token = None
        blank_idx = 0  # EMPTY_TOK is at index 0
        
        for token_idx in inds:
            if token_idx == blank_idx:
                continue  # Skip blank tokens
            if token_idx != prev_token:
                decoded.append(token_idx)
            prev_token = token_idx
        
        # ind2char
        text = "".join([self.ind2char[idx] for idx in decoded]).strip()
        
        return text

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
