from dataclasses import dataclass

import torch
from torch import nn

import torchaudio
import librosa

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 11025
    n_mels: int = 80
    power: float = 2.0

    # value of melspectrograms if silence
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    """
    Mel-spectrogram extractor compatible with WaveGlow/iSTFTNet.
    Uses Slaney mel scale
    """

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        # for Collab
        fb = torch.from_numpy(mel_basis.astype("float32")).to(
            self.mel_spectrogram.mel_scale.fb.dtype
        )
        self.mel_spectrogram.mel_scale.fb.copy_(fb)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T] or [T]
        :return: Shape is [B, n_mels, T'] or [n_mels, T']
        """

        # handle 1D input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # remove channel dimension if present [B, C, T] -> [B, T]
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
