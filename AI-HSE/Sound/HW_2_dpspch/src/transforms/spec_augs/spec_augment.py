import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpecAugment(nn.Module):
    """
    Applies frequency masking and time masking to spectrograms.
    Used only for training.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27, num_freq_masks: int = 2,      
        time_mask_ratio: float = 0.05, num_time_masks: int = 10,
    ):
        """
        Args:
            freq_mask_param (int): Maximum number of frequency channels to mask.
            num_freq_masks (int): Number of frequency masks to apply.
            time_mask_ratio (float): Maximum time-mask ratio pS.
            num_time_masks (int): Number of time masks to apply.
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_freq_masks = num_freq_masks
        self.time_mask_ratio = time_mask_ratio
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: Tensor) -> Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram (Tensor): Input spectrogram of shape [batch, n_feats, time]
        
        Returns:
            Tensor: Augmented spectrogram of shape [batch, n_feats, time]
        """
        # only train
        if not self.training:
            return spectrogram

        if spectrogram.requires_grad:
            aug_spec = spectrogram.clone()
        else:
            aug_spec = spectrogram
        
        _, n_feats, time_steps = aug_spec.shape
        
        # processing frequency masking
        for _ in range(self.num_freq_masks):
            aug_spec = self._apply_freq_mask(aug_spec, n_feats)
        
        # processing time masking
        for _ in range(self.num_time_masks):
            aug_spec = self._apply_time_mask(aug_spec, time_steps)
        
        return aug_spec
    
    def _apply_freq_mask(self, spec: Tensor, n_feats: int) -> Tensor:
        """
        Apply frequency masking: mask consecutive frequency channels.
        
        Args:
            spec (Tensor): Spectrogram of shape [batch, n_feats, time]
            n_feats (int): Number of frequency channels
        
        Returns:
            Tensor: Spectrogram with frequency masking applied
        """
        batch_size = spec.shape[0]
        
        # choose number of frequencies to mask
        freq = torch.randint(0, self.freq_mask_param + 1, size=(batch_size,), device=spec.device)
        
        # limitation of range
        freq = torch.clamp(freq, 0, n_feats)
        
        # random starting position for each sample in batch
        freq_0 = torch.zeros(batch_size, dtype=torch.long, device=spec.device)
        for i in range(batch_size):
            max_start = max(1, n_feats - freq[i].item())
            freq_0[i] = torch.randint(0, max_start, size=(1,), device=spec.device).item()
        
        # apply masking for each sample in batch
        for i in range(batch_size):
            if freq[i] > 0:
                spec[i, freq_0[i]:freq_0[i] + freq[i], :] = 0
        
        return spec
    
    def _apply_time_mask(self, spec: Tensor, time_steps: int) -> Tensor:
        """
        Apply time masking: mask consecutive time steps.
        
        Common approach:
        - Maximum time mask size = pS * utterance_length (pS = 0.05)
        - For each sample, max_mask_size = time_mask_ratio * time_steps
        
        Args:
            spec (Tensor): Spectrogram of shape [batch, n_feats, time]
            time_steps (int): Number of time steps
        
        Returns:
            Tensor: Spectrogram with time masking applied
        """
        batch_size = spec.shape[0]
        
        # max time mask size for each sample
        max_time_mask_size = int(self.time_mask_ratio * time_steps)
        
        # choose number of time steps to mask
        t = torch.randint(
            0, 
            max_time_mask_size + 1, 
            size=(batch_size,), 
            device=spec.device
        )
        
        # limitation of range
        t = torch.clamp(t, 0, time_steps)
        
        # random start position for each sample in batch
        t0 = torch.zeros(batch_size, dtype=torch.long, device=spec.device)
        for i in range(batch_size):
            max_start = max(1, time_steps - t[i].item())
            t0[i] = torch.randint(0, max_start, size=(1,), device=spec.device).item()
        
        # apply masking
        for i in range(batch_size):
            if t[i] > 0:
                spec[i, :, t0[i]:t0[i] + t[i]] = 0
        
        return spec

