import torch
import torchaudio
from torch import Tensor, nn


class SpeedPerturbation(nn.Module):
    """
    Speed Perturbation augmentation.
    Changes audio speed by resampling, which affects both speed and pitch.
    """
    
    def __init__(
        self,
        speeds: list = [0.9, 1.0, 1.1],
        sample_rate: int = 16000,
        p: float = 0.5,
    ):
        """
        Args:
            speeds (list): List of speed factors to choose from (speed < 1.0 - slows down, > 1.0 - speeds up)
            sample_rate (int): Sample rate of the audio.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.speeds = speeds
        self.sample_rate = sample_rate
        self.p = p
    
    def __call__(self, audio: Tensor) -> Tensor:
        """
        Apply speed perturbation to audio.
        
        Args:
            audio (Tensor): Audio waveform of shape [time] or [channels, time]
        
        Returns:
            Tensor: Augmented audio with same shape as input
        """
        if not self.training:
            return audio
        
        # where to apply augmentation
        if torch.rand(1).item() > self.p:
            return audio
        
        # random speed
        speed = self.speeds[torch.randint(0, len(self.speeds), (1,)).item()]
        
        # if speed is 1.0, no change needed
        if speed == 1.0:
            return audio
        
        # original shape
        original_shape = audio.shape
        original_length = audio.shape[-1]
        
        # channel dim
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, time]
        
        # calc new length
        new_length = int(original_length / speed)
        
        # add batch dim for interpolate
        audio_batch = audio.unsqueeze(0)  # [1, channels, time]
        
        # linear interpolation to change length
        audio_batch = torch.nn.functional.interpolate(
            audio_batch,
            size=new_length,
            mode='linear',
            align_corners=False
        )
        
        # remove batch dim
        audio = audio_batch.squeeze(0)
        
        # if new length is different, pad or crop to original length
        if new_length != original_length:
            if new_length > original_length:
                audio = audio[..., :original_length]
            else:
                padding = original_length - new_length
                audio = torch.nn.functional.pad(audio, (0, padding))
        
        # restore original shape
        if original_shape != audio.shape:
            audio = audio.reshape(original_shape)
        
        return audio

