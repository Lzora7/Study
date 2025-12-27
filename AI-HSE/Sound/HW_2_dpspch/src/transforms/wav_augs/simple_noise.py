import torch
from torch import Tensor, nn


class SimpleNoise(nn.Module):
    """
    Simple noise augmentation.
    Adds Gaussian or uniform noise to audio samples.
    """
    
    def __init__(
        self,
        noise_type: str = "gaussian",  # "gaussian" or "uniform"
        snr_range: tuple = (0, 15),  # Signal-to-Noise Ratio in dB
        p: float = 0.5,
    ):
        """
        Args:
            noise_type (str): Type of noise to add. "gaussian" or "uniform".
            snr_range (tuple): Range of SNR values (min, max) in dB (lower SNR = more noise).
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.noise_type = noise_type
        self.snr_range = snr_range
        self.p = p
    
    def __call__(self, audio: Tensor) -> Tensor:
        """
        Add simple noise to audio.
        
        Args:
            audio (Tensor): Audio waveform of shape [time] or [channels, time]
        
        Returns:
            Tensor: Audio with noise added, same shape as input
        """
        if not self.training:
            return audio
        
        # where to apply augmentation
        if torch.rand(1).item() > self.p:
            return audio
        
        original_shape = audio.shape
        
        # channel dim
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, time]
        
        # calce SNR and generate noise
        snr_db = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()
        snr_linear = 10 ** (snr_db / 10)
        
        # calc signal power
        signal_power = (audio ** 2).mean()
        
        # generate noise
        if self.noise_type == "gaussian":
            noise = torch.randn_like(audio)
        else:
            noise = torch.empty_like(audio).uniform_(-1, 1)

        # calc noise power
        noise_power = (noise ** 2).mean()
        
        # scale noise to desired SNR
        if noise_power > 0:
            scale_factor = torch.sqrt(signal_power / (noise_power * snr_linear))
            noise = noise * scale_factor
        else:
            noise = torch.zeros_like(audio)
        
        # ddd noise to audio
        audio = audio + noise
        
        # original shape
        if original_shape != audio.shape:
            audio = audio.reshape(original_shape)
        
        return audio

