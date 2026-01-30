from torch import nn
from torch.nn.utils import weight_norm

class UpsampleNet(nn.Module):
    """
    Upsample Network that upscale time dimension.
    Upsample goes step by step: for x8 it's: x2 -> x2 -> x2
    """

    def __init__(self, in_channels=80, out_channels=16):
        """
        Args:
            in_channels (int): amount of input channels
            out_channels (int): amount of output channels (FFT size for iSTFT, basically = 16)
        """
        super().__init__()
        
        # layer 1: ×2
        # [B, 80, T] → [B, 80, 2*T] → [B, 40, 2*T]
        self.nearest1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, 
            in_channels // 2,
            kernel_size=3, 
            stride=1, 
            padding=1
            )
        )
        
        # layer 2: ×2
        # [B, 40, 2*T] → [B, 40, 4*T] → [B, 20, 4*T]
        self.nearest2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = weight_norm(nn.Conv1d(
            in_channels // 2, 
            in_channels // 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
            )
        )
        
        # layer 3: ×2
        # [B, 20, 4*T] → [B, 20, 8*T] → [B, 16, 8*T]
        self.nearest3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = weight_norm(nn.Conv1d(
            in_channels // 4, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
            )
        )
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, mel_spec, **batch):
        """
        Model forward method.

        Args:
            mel_spec (Tensor): input mel-spectrogram [B, C, T]
        Returns:
            out (Tensor): upsampled spectrogram [B, out_channels, T*upsample_coef] (ready for iSTFT)
        """
        # layer 1
        out = self.nearest1(mel_spec)
        out = self.conv1(out)   
        out = self.leaky_relu(out)
        
        # layer 2
        out = self.nearest2(out)    
        out = self.conv2(out)         
        out = self.leaky_relu(out)
        
        # layer 3
        out = self.nearest3(out)
        out = self.conv3(out)        

        return out