from typing import List

import numpy as np
import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

from pyctcdecode import BeamSearchDecoderCTC, Alphabet


# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    """
    Calculate WER metric with Beam Search decoding
    """
    
    def __init__(self, text_encoder, beam_size=5, top_k_tokens=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.top_k_tokens = top_k_tokens  # Consider only top-k tokens at each step
        self.blank_idx = 0  # EMPTY_TOK is at index 0 (used in beam search CTC rules)

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Calculate WER using beam search decoding.
        
        Args:
            log_probs: Tensor of shape [batch, time, vocab] with log probabilities
            log_probs_length: Tensor of shape [batch] with actual sequence lengths
            text: List of target text strings
        """
        wers = []
        log_probs_np = log_probs.cpu().detach().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        
        for log_prob_seq, length, target_text in zip(log_probs_np, lengths, text):
            # cut sequence to actual length
            seq_log_probs = log_prob_seq[:int(length)]
            
            # beam search
            best_sequence = self._beam_search(seq_log_probs)
            
            # decode sequence with CTC rules
            pred_text = self.text_encoder.ctc_decode(best_sequence)
            
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        
        return sum(wers) / len(wers) if wers else 0.0

    def _beam_search(self, log_probs_seq):
        """
        Perform beam search decoding for a single sequence.
        
        Args:
            log_probs_seq: numpy array of shape [time, vocab] with log probabilities
            
        Returns:
            best_sequence: list of token indices (best hypothesis)
        """
        # Each hypothesis is (sequence, log_prob, last_char)
        
        # sequence - list of token indices (after CTC merge)
        # log_prob - cumulative log probability
        # last_char - last non-blank character (used for CTC merge rules)
        
        beam = [([], 0.0, None)]  # empty sequence
        
        for t in range(len(log_probs_seq)):
            time_log_probs = log_probs_seq[t]  # [vocab] in current t
            candidates = []
            
            # expand
            for sequence, log_prob, last_char in beam:
                # get top-k token indices by probability
                if self.top_k_tokens < len(time_log_probs):
                    top_k_indices = np.argsort(time_log_probs)[-self.top_k_tokens:][::-1]
                    token_indices = top_k_indices
                else:
                    # if top_k_tokens >= vocab_size, k - all tokens
                    token_indices = range(len(time_log_probs))
                
                # try top-k tokens at this time step
                for token_idx in token_indices:
                    token_log_prob = time_log_probs[token_idx]
                    new_log_prob = log_prob + token_log_prob
                    
                    # apply merge rules (CTC)
                    if token_idx == self.blank_idx or token_idx == last_char:
                        # blank token or same as last - don't add duplicate
                        candidates.append((sequence, new_log_prob, last_char))
                    else:
                        # new token - add to sequence
                        new_sequence = sequence + [token_idx]
                        candidates.append((new_sequence, new_log_prob, token_idx))
            
            # top beam_size hypotheses
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_size]
        
        # return best sequence
        best_sequence = max(beam, key=lambda x: x[1])[0]
        return best_sequence


class LibraryBeamSearchWERMetric(BaseMetric):
    """
    Calculate WER metric using library-based Beam Search decoding (pyctcdecode).
    This uses the pyctcdecode library for efficient CTC beam search.
    """
    
    def __init__(self, text_encoder, beam_size=5, blank_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.blank_idx = blank_idx
        
        labels = [text_encoder.EMPTY_TOK] + text_encoder.alphabet
        alphabet = Alphabet.build_alphabet(labels)
        
        self.decoder = BeamSearchDecoderCTC(alphabet, language_model=None)
    
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Calculate WER using library-based beam search decoding.
        
        Args:
            log_probs: Tensor of shape [batch, time, vocab] with log probabilities
            log_probs_length: Tensor of shape [batch] with actual sequence lengths
            text: List of target text strings
        """
        wers = []
        
        log_probs_np = log_probs.cpu().detach().numpy()
        lengths_np = log_probs_length.cpu().detach().numpy()
        
        for i, (log_prob_seq, length, target_text) in enumerate(zip(log_probs_np, lengths_np, text)):
            # Cut sequence to actual length: [time, vocab]
            seq_log_probs = log_prob_seq[:int(length)]
            
            # decode using pyctcdecode
            pred_text = self.decoder.decode(
                seq_log_probs,
                beam_width=self.beam_size,
                beam_prune_logp=-10.0,
                token_min_logp=-5.0
            )
            
            target_text = self.text_encoder.normalize_text(target_text)
            
            wers.append(calc_wer(target_text, pred_text))
        
        return sum(wers) / len(wers) if wers else 0.0
