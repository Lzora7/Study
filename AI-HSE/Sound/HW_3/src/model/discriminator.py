from torch import nn
from .dis_blocks.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from .dis_blocks.MultiScaleDiscriminator import MultiScaleDiscriminator


class HiFiGANDiscriminator(nn.Module):
    """
    HiFi-GAN Discriminator combining MPD and MSD
    """

    def __init__(self, periods=[2, 3, 5, 7, 11]):
        """
        Args:
            periods (list): list of periods for MPD
        """
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T] or [B, T]
        Returns:
            mpd_outputs (list): outputs from Multi Period Discriminator
            msd_outputs (list): outputs from Multi Scale Discriminator
            mpd_features (list): features from MPD for feature matching
            msd_features (list): features from MSD for feature matching
        """
        # valid input is [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # mpd
        mpd_outputs, mpd_features = self.mpd(x)
        
        # msd
        msd_outputs, msd_features = self.msd(x)
        
        return {
            "mpd_outputs": mpd_outputs,
            "msd_outputs": msd_outputs,
            "mpd_features": mpd_features,
            "msd_features": msd_features,
        }
