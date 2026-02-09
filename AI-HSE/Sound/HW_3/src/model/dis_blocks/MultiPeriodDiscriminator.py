from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class PeriodDiscriminator(nn.Module):
    """
    Period Discriminator from HiFi-GAN.
    Discriminates based on different periods of the waveform.
    """

    def __init__(self, period, kernel_size=5, stride=3):
        """
        Args:
            period (int): Period to check (e.g., 2, 3, 5, 7, 11)
            kernel_size (int): Kernel size for conv
            stride (int): Stride for conv
        """
        super().__init__()
        self.period = period
        
        # stack of convs
        layers = []
        in_channels = 1
        
        # multiple conv layers with increasing channels
        for i in range(4):
            out_channels = min(1024, 2 ** (i + 2))
            layers.append(
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0)
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            in_channels = out_channels
        
        # final conv layer
        layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    1,
                    (3, 1),
                    padding=(1, 0)
                )
            )
        )
        
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T]
        Returns:
            output (Tensor): Discriminator output [B, 1, T']
            features (list): List of features for feature matching
        """
        # reshape to [B, 1, period, T//period]
        length = x.shape[-1]
        if length % self.period != 0:
            pad_len = self.period - (length % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
        
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, self.period, length // self.period)
        
        features = []
        for layer in self.convs[:-1]:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        
        # final layer
        x = self.convs[-1](x)
        
        # reshape back to [B, 1, T']
        x = x.view(batch_size, 1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) from HiFi-GAN.
    Uses multiple PeriodDiscriminators with different periods.
    """

    def __init__(self, periods=[2, 3, 5, 7, 11]):
        """
        Args:
            periods (list): List of periods to use
        """
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T]
        Returns:
            outputs (list): List of discriminator outputs
            all_features (list): List of all features
        """
        outputs = []
        all_features = []
        
        for disc in self.discriminators:
            output, features = disc(x)
            outputs.append(output)
            all_features.extend(features)
        
        return outputs, all_features