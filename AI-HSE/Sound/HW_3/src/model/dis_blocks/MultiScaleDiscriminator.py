from torch import nn
from torch.nn.utils import weight_norm

class ScaleDiscriminator(nn.Module):
    """
    Scale Discriminator from HiFi-GAN.
    Discriminates at different scales (downsampled versions).
    """

    def __init__(self):
        super().__init__()
        
        # Stack of convolutions
        layers = []
        in_channels = 1
        
        # Multiple conv layers with increasing channels
        for i in range(4):
            out_channels = min(1024, 2 ** (i + 2))
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=15,
                        stride=1,
                        padding=7
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            
            # Downsample
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size=41,
                        stride=4,
                        groups=out_channels,
                        padding=20
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            in_channels = out_channels
        
        # Final conv layer
        layers.append(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1
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
            features (list): List of intermediate features for feature matching
        """
        features = []
        for i, layer in enumerate(self.convs[:-1]):
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and i % 2 == 1:
                features.append(x)
        
        # Final layer
        x = self.convs[-1](x)
        
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) from HiFi-GAN.
    Uses multiple ScaleDiscriminators at different scales.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(3)
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform [B, 1, T]
        Returns:
            outputs (list): List of discriminator outputs at different scales
            all_features (list): List of all features from all discriminators
        """
        outputs = []
        all_features = []
        
        # Original scale
        output, features = self.discriminators[0](x)
        outputs.append(output)
        all_features.extend(features)
        
        # Downsampled scales (cascade: each scale is pooled from the previous)
        x_cur = x
        for i, pool in enumerate(self.pooling):
            x_cur = pool(x_cur)
            output, features = self.discriminators[i + 1](x_cur)
            outputs.append(output)
            all_features.extend(features)
        
        return outputs, all_features