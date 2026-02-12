import torch
from torch import nn
from torch.nn.utils import weight_norm
from .blocks.ResBlock import ResBlock

def get_padding(kernel_size, dilation=1):
    """Calculate padding for Conv1d to maintain same length."""
    return int((kernel_size * dilation - dilation) / 2)

class iSTFTNET_model(nn.Module):
    """
    iSTFTNet model with V1-C8C8C2I architecture:
    - C8: Upsample ×8 (512->256) + 8 ResBlocks (256 channels)
    - C8: Upsample ×8 (256->128) + 8 ResBlocks (128 channels)
    - C2: Upsample ×2 (128->64) + 2 ResBlocks (64 channels)
    - C2: Upsample ×2 (64->32) + 2 ResBlocks (32 channels)
    - I: iSTFT
    """

    def __init__(self, in_channels=80, upsample_initial_channel=512, n_fft=16, hop_length=None, win_length=None):
        """
        Args:
            in_channels (int): number of input mel-spectrogram channels (usually 80)
            upsample_initial_channel (int): initial channel size (512 for V1)
            n_fft (int): FFT size for iSTFT (usually 16)
            hop_length (int): hop length for iSTFT (default: n_fft // 4)
            win_length (int): window length for iSTFT (default: n_fft)
        """
        super().__init__()
        
        self.n_fft = n_fft

        # default values
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        
        # ensure win_length <= n_fft
        if self.win_length > self.n_fft:
            self.win_length = self.n_fft
        
        # Input conv (mel-spectrogram -> features)
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        
        # Upsampling layers and ResBlocks

        # C8: Upsample ×8: 512 -> 256
        self.ups_1 = weight_norm(nn.ConvTranspose1d(
            upsample_initial_channel // (2**0),  # 512
            upsample_initial_channel // (2**1),  # 256
            kernel_size=16,
            stride=8,
            padding=(16-8)//2
        ))
        self.resblocks_1 = nn.ModuleList([
            ResBlock(ch=upsample_initial_channel // (2**1)) for _ in range(8)  # 256 channels
        ])
        
        # C8: Upsample ×8: 256 -> 128
        self.ups_2 = weight_norm(nn.ConvTranspose1d(
            upsample_initial_channel // (2**1),  # 256
            upsample_initial_channel // (2**2),  # 128
            kernel_size=16,
            stride=8,
            padding=(16-8)//2
        ))
        self.resblocks_2 = nn.ModuleList([
            ResBlock(ch=upsample_initial_channel // (2**2)) for _ in range(8)  # 128 channels
        ])
        
        # C2: Upsample ×2: 128 -> 64
        self.ups_3 = weight_norm(nn.ConvTranspose1d(
            upsample_initial_channel // (2**2),  # 128
            upsample_initial_channel // (2**3),  # 64
            kernel_size=4,
            stride=2,
            padding=(4-2)//2
        ))
        self.resblocks_3 = nn.ModuleList([
            ResBlock(ch=upsample_initial_channel // (2**3)) for _ in range(2)  # 64 channels
        ])
        
        # C2: Upsample ×2: 64 -> 32
        self.ups_4 = weight_norm(nn.ConvTranspose1d(
            upsample_initial_channel // (2**3),  # 64
            n_fft,  # 16
            kernel_size=4,
            stride=2,
            padding=(4-2)//2
        ))
        
        # Output conv
        n_bins = n_fft // 2 + 1
        self.conv_post = weight_norm(nn.Conv1d(n_fft, n_bins * 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

    def forward(self, mel_spec, **batch):
        """
        Model forward method matching original iSTFTNet.

        Args:
            mel_spec (Tensor): mel-spectrogram [B, C, T]
                B - Batch
                C - Channels
                T - Time
        Returns:
            waveform (Tensor): output waveform [B, L]
                B - Batch
                L - Length of waveform
        """
        # Input conv
        x = self.conv_pre(mel_spec)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        
        # C8: 1st upsampling + ResBlocks
        x = self.ups_1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        for res_block in self.resblocks_1:
            x = res_block(x)
        
        # C8: 2nd upsampling + ResBlocks
        x = self.ups_2(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        for res_block in self.resblocks_2:
            x = res_block(x)
        
        # C2: 3rd upsampling + ResBlocks
        x = self.ups_3(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        for res_block in self.resblocks_3:
            x = res_block(x)
        
        # C2: 4th upsampling (to n_fft channels)
        x = self.ups_4(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        
        # Output conv
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        
        # Split into magnitude and phase
        n_bins = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :n_bins, :])  
        phase = torch.sin(x[:, n_bins:, :])
        
        # convert to complex spectrogram
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        complex_spec = real_part + 1j * imag_part
        
        # convert to waveform
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(complex_spec.device),
            center=True,
            normalized=False,
            onesided=True,
            length=None
        )

        return waveform

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths * 256

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
