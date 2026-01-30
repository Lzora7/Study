import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .ConvBlock import ConvBlock

class ResBlock(nn.Module):
    """
    Residual Block with Multi-Receptive Field (MRF) feature extraction.
    """

    def __init__(self, ch, kernel_sizes=[3,7,11], dilations=[1,3,5]):
        """
        Args:
            ch (int): number of channels.
            kernel_sizes (List[int]): list of kernel sizes for each ConvBlock.
            dilations (List[int]): list of dilations for each ConvBlock.
        """
        super().__init__()

        # MRF blocks
        self.conv_block_1 = ConvBlock(ch=ch, kernel_sizes=kernel_sizes, dilations=dilations)
        self.conv_block_2 = ConvBlock(ch=ch, kernel_sizes=kernel_sizes, dilations=dilations)
        self.conv_block_3 = ConvBlock(ch=ch, kernel_sizes=kernel_sizes, dilations=dilations)
        
        # MRF feature extraction
        self.mrf_extraction = nn.Conv1d(ch*3, ch, kernel_size=1)

    def forward(self, mel_spec, **batch):
        """
        Model forward method.

        Args:
            mel_spec (Tensor): input spectrogram [B, C, T]
        Returns:
            out (Tensor): output spectrogram [B, C, T] (same size)
        """
        # feature extraction
        f1 = self.conv_block_1(mel_spec)
        f2 = self.conv_block_2(mel_spec)
        f3 = self.conv_block_3(mel_spec)

        # concat features
        f_concat = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C, T]

        # MRF extraction
        mrf = weight_norm(self.mrf_extraction(f_concat))  # [B, C, T]
        mrf = F.leaky_relu(mrf, 0.1)

        # Residual connection
        return mrf + mel_spec


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
