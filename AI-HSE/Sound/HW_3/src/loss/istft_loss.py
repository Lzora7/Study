from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import torchaudio


class iSTFTLoss(Module):
    """
    Loss function for iSTFTNet.

    Generator loss:
    1. Mel-spectrogram loss (L1 or L2 on log-mel)
    2. Adversarial loss (LSGAN): sum over all sub-discriminators
    3. Feature matching loss: 2 * sum over all layers

    Discriminator loss:
    Sum over all sub-discriminators
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = None,
        mel_loss_type: str = "l1",  # "l1" or "l2"
        mel_loss_weight: float = 45.0,
        adv_loss_weight: float = 1.0,
        fm_loss_weight: float = 2.0,
    ):
        """
        Args:
            sample_rate (int): Sample rate of audio
            n_fft (int): FFT size for mel-spectrogram
            win_length (int): Window length for mel-spectrogram
            hop_length (int): Hop length for mel-spectrogram
            n_mels (int): Number of mel filter banks
            f_min (float): Minimum frequency
            f_max (float): Maximum frequency (None = sample_rate / 2)
            mel_loss_type (str): Type of mel loss - "l1" or "l2"
            mel_loss_weight (float): Weight for mel-spectrogram loss
            adv_loss_weight (float): Weight for adversarial loss (0 to disable)
            fm_loss_weight (float): Weight for feature matching loss (0 to disable)
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.mel_loss_type = mel_loss_type
        
        # Loss weights
        self.mel_loss_weight = mel_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.fm_loss_weight = fm_loss_weight
        
        # Create mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
        )

    def forward(
        self,
        waveform_pred: Tensor,
        waveform_target: Tensor,
        discriminator_outputs: list = None,
        discriminator_features_pred: list = None,
        discriminator_features_target: list = None,
        **batch
    ) -> dict:
        """
        Generator loss L_G: mel + adversarial (LSGAN) + feature matching.

        Args:
            waveform_pred (Tensor): Predicted waveform [B, L]
            waveform_target (Tensor): Target waveform [B, L]
            discriminator_outputs (list of Tensor, optional): List of outputs from all
                sub-discriminators (MPD + MSD) for predicted waveform. Each can be [B, 1, T'] or [B, N].
            discriminator_features_pred (list, optional): List of intermediate features for predicted waveform
            discriminator_features_target (list, optional): List of intermediate features for target waveform
            **batch: Additional batch data

        Returns:
            dict: loss, mel_loss, adv_loss (if used), fm_loss (if used)
        """
        # Ensure waveforms have the same length
        min_length = min(waveform_pred.shape[-1], waveform_target.shape[-1])
        waveform_pred = waveform_pred[..., :min_length]
        waveform_target = waveform_target[..., :min_length]

        # 1. Mel-spectrogram loss (as in HiFi-GAN / iSTFTNet)
        mel_pred = self.mel_transform(waveform_pred)  # [B, n_mels, T]
        mel_target = self.mel_transform(waveform_target)
        mel_pred_log = torch.log(mel_pred + 1e-5)
        mel_target_log = torch.log(mel_target + 1e-5)
        if self.mel_loss_type == "l1":
            mel_loss = F.l1_loss(mel_pred_log, mel_target_log)
        elif self.mel_loss_type == "l2":
            mel_loss = F.mse_loss(mel_pred_log, mel_target_log)
        else:
            raise ValueError(f"Unknown mel_loss_type: {self.mel_loss_type}. Use 'l1' or 'l2'")

        total_loss = self.mel_loss_weight * mel_loss
        losses = {"mel_loss": mel_loss}

        # 2. Adversarial loss (LSGAN): sum over all sub-discriminators of E[(1 - D(y_hat))^2]
        if discriminator_outputs is not None and self.adv_loss_weight > 0:
            adv_loss = torch.tensor(0.0, device=waveform_pred.device, dtype=waveform_pred.dtype)
            for dg in discriminator_outputs:
                dg_flat = dg.reshape(dg.size(0), -1)
                adv_loss += torch.mean((1 - dg_flat) ** 2)
            total_loss += self.adv_loss_weight * adv_loss
            losses["adv_loss"] = adv_loss

        # 3. Feature matching: 2 * sum over all layers of E[|f_real - f_fake|] (as in HiFi-GAN)
        if (
            discriminator_features_pred is not None
            and discriminator_features_target is not None
            and self.fm_loss_weight > 0
        ):
            fm_loss = torch.tensor(0.0, device=waveform_pred.device, dtype=waveform_pred.dtype)
            for feat_pred, feat_target in zip(
                discriminator_features_pred, discriminator_features_target
            ):
                if feat_pred.shape != feat_target.shape:
                    min_shape = tuple(
                        min(sp, st) for sp, st in zip(feat_pred.shape, feat_target.shape)
                    )
                    feat_pred = feat_pred[tuple(slice(0, s) for s in min_shape)]
                    feat_target = feat_target[tuple(slice(0, s) for s in min_shape)]
                fm_loss += torch.mean(torch.abs(feat_pred - feat_target))
            fm_loss = fm_loss * 2
            total_loss += self.fm_loss_weight * fm_loss
            losses["fm_loss"] = fm_loss

        losses["loss"] = total_loss
        return losses

    @staticmethod
    def discriminator_loss(
        disc_real_outputs: List[Tensor],
        disc_pred_outputs: List[Tensor],
    ) -> Tuple[Tensor, List[float], List[float]]:
        """
        Discriminator loss L_D (LSGAN): sum over all sub-D of E[(1-D(y))^2] + E[D(y_hat)^2].

        Args:
            disc_real_outputs (list of Tensor): Outputs of all sub-D on real waveform
            disc_pred_outputs (list of Tensor): Outputs of all sub-D on predicted waveform

        Returns:
            loss (Tensor): Total discriminator loss
            r_losses (list): Per-sub-discriminator loss on real
            g_losses (list): Per-sub-discriminator loss on fake
        """
        loss = torch.tensor(0.0, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_pred_outputs):
            dr_flat = dr.reshape(dr.size(0), -1)
            dg_flat = dg.reshape(dg.size(0), -1)
            r_loss = torch.mean((1 - dr_flat) ** 2)
            g_loss = torch.mean(dg_flat ** 2)
            loss = loss + r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses
