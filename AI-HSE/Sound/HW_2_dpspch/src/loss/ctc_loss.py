import torch
from torch import Tensor
from torch.nn import CTCLoss
import logging

logger = logging.getLogger(__name__)


class CTCLossWrapper(CTCLoss):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True, **kwargs):
        """
        Initialize CTC Loss with proper parameters.
        
        Args:
            blank (int): blank label index (default: 0).
            reduction (str): reduction method - 'mean', 'sum', or 'none' (default: 'mean').
            zero_infinity (bool): if True, replaces infinite loss with zero (default: True).
        """
        super().__init__(blank=blank, reduction=reduction, zero_infinity=zero_infinity, **kwargs)
        logger.info(f"CTCLossWrapper initialized with blank={blank}, reduction={reduction}, zero_infinity={zero_infinity}")
    
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:

        # move to CPU
        if log_probs.device.type == 'mps':
            log_probs = log_probs.cpu()
            log_probs_length = log_probs_length.cpu()
            text_encoded = text_encoded.cpu()
            text_encoded_length = text_encoded_length.cpu()
        
        # check input_lengths >= target_lengths (else it's not valid for CTC)
        valid_mask = log_probs_length >= text_encoded_length
        
        if not valid_mask.all():
            if valid_mask.any():
                # CTC for valid examples
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                log_probs_valid = log_probs[valid_indices]
                log_probs_length_valid = log_probs_length[valid_indices]
                text_encoded_valid = text_encoded[valid_indices]
                text_encoded_length_valid = text_encoded_length[valid_indices]
                
                log_probs_t = torch.transpose(log_probs_valid, 0, 1)
                
                loss = super().forward(
                    log_probs=log_probs_t,
                    targets=text_encoded_valid,
                    input_lengths=log_probs_length_valid,
                    target_lengths=text_encoded_length_valid,
                )
            else:
                # everything is not valid - return big loss
                logger.error("All examples in batch are invalid! Returning large loss.")
                loss = torch.tensor(1e6, device=log_probs.device, dtype=log_probs.dtype)
        else:
            # everything is valid - simple forward
            log_probs_t = torch.transpose(log_probs, 0, 1)

            loss = super().forward(
                log_probs=log_probs_t,
                targets=text_encoded,
                input_lengths=log_probs_length,
                target_lengths=text_encoded_length,
            )
        
        # back to main device
        if log_probs.device.type == 'mps':
            loss = loss.to(log_probs.device)

        return {"loss": loss}
