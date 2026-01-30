import torch
from torch import nn
import torch.nn.functional as F
from blocks import ResBlock, UpsampleNet

class iSTFTNET_model(nn.Module):
    """
    iSTFTNet model with V1-C8C8C2I architecture:
    - C8: 8 ResBlocks (first group)
    - C8: 8 ResBlocks (second group)
    - C2: UpsampleNet (2 upsampling layers, ×8 total)
    - I: iSTFT
    """

    def __init__(self, in_channels=80, n_features=512, n_fft=16, hop_length=256, win_length=1024):
        """
        Args:
            in_channels (int): number of input mel-spectrogram channels (usually 80)
            n_features (int): number of hidden features (usually 512)
            n_fft (int): FFT size for iSTFT (usually 16)
            hop_length (int): hop length for iSTFT
            win_length (int): window length for iSTFT
        """
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # input Conv (mel-spectrogram -> features)
        self.InputConv = nn.Conv1d(
            in_channels, 
            n_features, 
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.LeakyRelu = nn.LeakyReLU(0.1)

        # C8: first group of 8 ResBlocks
        self.res_blocks_1 = nn.ModuleList([
            ResBlock(ch=n_features) for _ in range(8)
        ])
        
        # C8: second group of 8 ResBlocks
        self.res_blocks_2 = nn.ModuleList([
            ResBlock(ch=n_features) for _ in range(8)
        ])
        
        # C2: UpsampleNet (×8 upsampling)
        # [B, n_features, T] -> [B, n_fft, 8*T]
        self.UpsampleNet = UpsampleNet(
            in_channels=n_features, 
            out_channels=n_fft
        )
        
        # output Conv (prepare for iSTFT)
        n_bins = n_fft // 2 + 1  # number of frequency bins
        self.OutputConv = nn.Conv1d(
            n_fft,
            2 * n_bins,  # magnitude + phase
            kernel_size=7,
            stride=1,
            padding=3
        )

    def forward(self, mel_spec, **batch):
        """
        Model forward method.

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
        # Input Conv
        out = self.LeakyRelu(self.InputConv(mel_spec))
        
        # C8: First group of 8 ResBlocks
        for res_block in self.res_blocks_1:
            out = res_block(out)
        
        # C8: Second group of 8 ResBlocks
        for res_block in self.res_blocks_2:
            out = res_block(out)
        
        # C2: UpsampleNet (×8 upsampling)
        out = self.UpsampleNet(out)  # [B, n_fft, 8*T]
        
        # Output Conv
        out = self.OutputConv(out)  # [B, 2*n_bins, 8*T]
        
        # split into magnitude and phase
        n_bins = self.n_fft // 2 + 1
        magnitude = torch.exp(out[:, :n_bins])  # [B, n_bins, 8*T]
        phase = out[:, n_bins:]  # [B, n_bins, 8*T]
        
        # convert phase to real and imaginary parts
        real_part = torch.cos(phase)
        imag_part = torch.sin(phase)
        
        # complex spec
        complex_spec = magnitude * (real_part + 1j * imag_part)  # [B, n_bins, 8*T]
        
        # iSTFT: convert to waveform
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(complex_spec.device),
            center=True,
            normalized=False,
            onesided=True,
            length=None  # Let it compute automatically
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
        return input_lengths  # we don't reduce time dimension here

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
